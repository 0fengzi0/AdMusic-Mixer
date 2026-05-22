#!/usr/bin/env python3
"""
广告混音 API 服务 - api_server.py
提供 RESTful API 接口用于在线混音处理。
启动: uvicorn api_server:app --host 0.0.0.0 --port 8000
"""

import json
import logging
import subprocess
import tempfile
import uuid
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import imageio_ffmpeg
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("api_ad_mix")

app = FastAPI(
    title="广告混音 API",
    description="自动分析歌曲特征 → 找最佳广告插入点 → 混音+淡出+裁剪",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 临时文件目录
TEMP_DIR = Path(tempfile.gettempdir()) / "ad_mix_api"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
#  核心算法（与 batch_ad_mix.py 一致）
# ═══════════════════════════════════════════════════════════════════

def analyze_audio(audio_path: str, sr: int = 22050) -> dict:
    """分析音频，返回 RMS 能量等信息"""
    log.info(f"分析音频: {audio_path}")
    cmd = [
        FFMPEG_BIN, "-v", "error", "-i", audio_path,
        "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"FFmpeg 解码失败: {stderr[-500:]}")

    y = np.frombuffer(result.stdout, dtype=np.float32)
    duration = len(y) / sr

    hop_length = 512
    frame_length = 2048
    if len(y) < frame_length:
        y = np.pad(y, (0, frame_length - len(y)))

    frame_count = max(1, 1 + (len(y) - frame_length) // hop_length)
    starts = np.arange(frame_count) * hop_length
    rms = np.empty(frame_count, dtype=np.float32)
    for i, start in enumerate(starts):
        frame = y[start:start + frame_length]
        rms[i] = np.sqrt(np.mean(frame * frame))
    times = starts / sr

    window_size = max(1, int(sr / hop_length * 1.0))
    rms_smooth = np.convolve(rms, np.ones(window_size) / window_size, mode="same")

    log.info(f"  总时长 {duration:.1f}s，帧数 {len(rms)}")
    return {
        "duration": float(duration),
        "rms": rms,
        "rms_times": times,
        "rms_smooth": rms_smooth,
        "sr": sr,
        "hop": hop_length,
    }


def find_best_insertion_point(analysis: dict, search_start_sec: float,
                              search_end_sec: float, ad_duration: float,
                              target_duration: float, fade_duration: float) -> float:
    """在指定秒数范围内找低能量、变化平稳的广告插入点"""
    rms = analysis["rms_smooth"]
    times = analysis["rms_times"]
    total = analysis["duration"]

    start_time = search_start_sec
    end_time = search_end_sec

    mask = (times >= start_time) & (times <= end_time)
    if not mask.any():
        log.warning("  搜索范围为空，使用中间点")
        return total * 0.5

    segment_rms = rms[mask]
    segment_times = times[mask]

    rms_min, rms_max = segment_rms.min(), segment_rms.max()
    if rms_max > rms_min:
        rms_norm = (segment_rms - rms_min) / (rms_max - rms_min)
    else:
        rms_norm = np.zeros_like(segment_rms)

    energy_score = 1.0 - rms_norm

    diff = np.abs(np.diff(segment_rms))
    diff_pad = np.concatenate([[0], diff])
    stability_score = 1.0 - diff_pad / (diff_pad.max() + 1e-9)

    combined = 0.6 * energy_score + 0.4 * stability_score

    margin = 5.0
    available_end = total - ad_duration - margin
    available_end = min(available_end, target_duration - ad_duration - fade_duration - 1.0)
    avail_mask = segment_times <= available_end
    if not avail_mask.any():
        return max(0.0, available_end)

    combined_masked = np.where(avail_mask, combined, -np.inf)
    best_idx = int(np.argmax(combined_masked))
    best_time = float(segment_times[best_idx])

    log.info(f"  搜索范围: {start_time:.1f}s ~ {end_time:.1f}s")
    log.info(f"  最佳插入点: {best_time:.2f}s")
    return best_time


def find_best_end_point(analysis: dict, target_duration: float,
                       ad_start: float, ad_duration: float,
                       fade_duration: float, search_margin: float = 10.0,
                       max_duration_ratio: float = 0.6) -> float:
    """在目标时长附近找最佳结尾点"""
    rms = analysis["rms_smooth"]
    times = analysis["rms_times"]
    total = analysis["duration"]

    ad_end = ad_start + ad_duration
    min_music_after_ad = fade_duration + 1.0
    min_end = ad_end + min_music_after_ad
    max_end = max(min_end, total * max_duration_ratio - fade_duration)

    search_start = max(min_end, ad_end + 15.0)
    search_end = min(total - fade_duration, max_end,
                     max(target_duration + search_margin, search_start + 20.0))

    if search_end <= search_start:
        log.warning(f"  结尾搜索范围不足，使用最小值: {min_end:.1f}s")
        return min(min_end, total - fade_duration)

    mask = (times >= search_start) & (times <= search_end)
    if not mask.any():
        return min(target_duration, total - fade_duration)

    segment_rms = rms[mask]
    segment_times = times[mask]

    backward_diff_forward = -np.diff(
        np.concatenate([segment_rms, [segment_rms[-1]]]))[::-1]

    rms_min_v, rms_max_v = segment_rms.min(), segment_rms.max()
    if rms_max_v > rms_min_v:
        rms_norm = (segment_rms - rms_min_v) / (rms_max_v - rms_min_v)
    else:
        rms_norm = np.zeros_like(segment_rms)
    low_energy_score = 1.0 - rms_norm

    forward_diff = np.abs(np.diff(np.concatenate([[segment_rms[0]], segment_rms])))
    stability_score = 1.0 - forward_diff / (forward_diff.max() + 1e-9)

    local_window = max(3, int(2.0 / (analysis["hop"] / analysis["sr"])))
    valley_score = np.zeros_like(segment_rms)
    for i in range(len(segment_rms)):
        left = max(0, i - local_window)
        right = min(len(segment_rms), i + local_window + 1)
        local_min = segment_rms[left:right].min()
        local_max = segment_rms[left:right].max()
        if local_max > local_min:
            valley_score[i] = 1.0 - (segment_rms[i] - local_min) / (local_max - local_min)

    decline_score = backward_diff_forward / (backward_diff_forward.max() + 1e-9)
    combined = (0.35 * low_energy_score + 0.25 * stability_score +
                0.25 * decline_score + 0.15 * valley_score)

    over_target_penalty = np.maximum(segment_times - target_duration, 0) / max(search_margin, 1.0)
    combined = combined - 0.05 * over_target_penalty

    best_idx = int(np.argmax(combined))
    best_time = float(segment_times[best_idx])

    final_end = min(best_time, total - fade_duration, max_end)

    log.info(f"  结尾搜索: {search_start:.1f}s ~ {search_end:.1f}s")
    log.info(f"  最佳结尾点: {final_end:.2f}s")
    return final_end


def build_ffmpeg_filter(music_duration: float, ad_duration: float,
                        ad_start: float, target_duration: float,
                        duck_volume: float, fade_duration: float,
                        ad_volume: float) -> tuple:
    """构建 ffmpeg filter_complex 字符串"""
    ad_end = ad_start + ad_duration
    fade_start = target_duration - fade_duration

    duck_start = max(ad_start - 0.5, 0.0)
    duck_end = min(ad_end + 0.5, target_duration)

    music_vol = (
        f"if(between(t,{duck_start:.3f},{ad_start:.3f}),"
        f"{duck_volume:.2f}+(1-{duck_volume:.2f})*(1-(t-{duck_start:.3f})/{max(ad_start-duck_start,0.001):.3f}),"
        f"if(between(t,{ad_start:.3f},{ad_end:.3f}),"
        f"{duck_volume:.2f},"
        f"if(between(t,{ad_end:.3f},{duck_end:.3f}),"
        f"{duck_volume:.2f}+(1-{duck_volume:.2f})*(t-{ad_end:.3f})/{max(duck_end-ad_end,0.001):.3f},"
        f"1)))"
    )

    final_vol = f"{music_vol}*if(between(t,{fade_start:.3f},{target_duration:.3f}),1-(t-{fade_start:.3f})/{fade_duration:.3f},1)"

    filter_complex = (
        f"[0:a]volume='{final_vol}':eval=frame,atrim=0:{target_duration},asetpts=PTS-STARTPTS[music];"
        f"[1:a]volume={ad_volume:.2f},adelay={int(ad_start*1000)}|{int(ad_start*1000)}[ad];"
        f"[music][ad]amix=inputs=2:duration=first:normalize=0[aout]"
    )

    return filter_complex, "[aout]"


def mix_with_ffmpeg(music_path: str, ad_path: str, output_path: str,
                    target_duration: float, ad_start: float,
                    ad_duration: float, duck_volume: float,
                    fade_duration: float, ad_volume: float) -> bool:
    """调用 ffmpeg 执行混音"""
    filter_complex, output_label = build_ffmpeg_filter(
        music_duration=target_duration,
        ad_duration=ad_duration,
        ad_start=ad_start,
        target_duration=target_duration,
        duck_volume=duck_volume,
        fade_duration=fade_duration,
        ad_volume=ad_volume,
    )

    cmd = [
        FFMPEG_BIN, "-y",
        "-i", music_path,
        "-i", ad_path,
        "-filter_complex", filter_complex,
        "-map", output_label,
        "-t", str(target_duration),
        "-c:a", "libmp3lame",
        "-b:a", "192k",
        "-ar", "44100",
        "-ac", "2",
        output_path,
    ]

    log.info(f"  执行混音 ...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=300)
        if result.returncode == 0:
            log.info(f"  混音成功: {output_path}")
            return True
        else:
            log.error(f"  FFmpeg 失败: {result.stderr[-500:]}")
            return False
    except subprocess.TimeoutExpired:
        log.error("  FFmpeg 超时")
        return False


def get_duration_ffprobe(audio_path: str) -> float:
    """用 ffprobe 获取音频时长"""
    cmd = [
        FFMPEG_BIN.replace("ffmpeg", "ffprobe"),
        "-v", "error", "-show_entries", "format=duration",
        "-of", "json", audio_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=30)
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except Exception:
        analysis = analyze_audio(audio_path)
        return analysis["duration"]


# ═══════════════════════════════════════════════════════════════════
#  API 路由
# ═══════════════════════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "service": "ad-music-mixer"}


@app.post("/mix")
async def mix_audio(
    music: UploadFile = File(..., description="歌曲音频文件 (mp3/wav/flac/m4a)"),
    ad: UploadFile = File(..., description="广告音频文件 (mp3/wav/flac/m4a)"),
    target_duration: int = Form(120, description="目标成品时长（秒）"),
    duck_volume: float = Form(0.3, description="广告期间背景音乐音量 (0.0~1.0)"),
    ad_volume: float = Form(2.3, description="广告音量倍数"),
    fade_duration: float = Form(3.0, description="结尾淡出时长（秒）"),
    search_start_sec: float = Form(20.0, description="广告插入点搜索起点（秒）"),
    search_end_sec: float = Form(40.0, description="广告插入点搜索终点（秒）"),
    end_search_margin: float = Form(60.0, description="结尾裁剪搜索范围秒数"),
    max_duration_ratio: float = Form(0.6, description="成品最长占原歌曲比例"),
) -> FileResponse:
    """
    核心接口：上传音乐和广告，返回混音后的音频文件。

    流程: 分析音乐 → 找最佳插入点 → 找最佳结尾点 → 混音 → 返回文件
    """
    # 参数校验
    allowed_exts = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}
    music_ext = Path(music.filename or "music.mp3").suffix.lower()
    ad_ext = Path(ad.filename or "ad.mp3").suffix.lower()
    if music_ext not in allowed_exts:
        raise HTTPException(400, f"不支持的歌曲格式: {music_ext}")
    if ad_ext not in allowed_exts:
        raise HTTPException(400, f"不支持的广告格式: {ad_ext}")

    session_id = uuid.uuid4().hex[:8]
    work_dir = TEMP_DIR / session_id
    work_dir.mkdir(parents=True, exist_ok=True)

    music_path = work_dir / f"music_{session_id}{music_ext}"
    ad_path = work_dir / f"ad_{session_id}{ad_ext}"
    output_path = work_dir / f"output_{session_id}.mp3"

    try:
        # 保存上传文件
        with open(music_path, "wb") as f:
            f.write(await music.read())
        with open(ad_path, "wb") as f:
            f.write(await ad.read())

        log.info(f"收到请求 {session_id}: {music.filename} + {ad.filename}")

        # ① 获取时长
        music_duration = get_duration_ffprobe(str(music_path))
        ad_duration = get_duration_ffprobe(str(ad_path))
        log.info(f"  歌曲 {music_duration:.1f}s, 广告 {ad_duration:.1f}s")

        # ② 分析 + 找点
        analysis = analyze_audio(str(music_path))
        ad_start = find_best_insertion_point(
            analysis, search_start_sec, search_end_sec,
            ad_duration, target_duration, fade_duration,
        )

        # ③ 找结尾点
        best_end_time = find_best_end_point(
            analysis, target_duration, ad_start, ad_duration,
            fade_duration, end_search_margin, max_duration_ratio,
        )

        ad_end = ad_start + ad_duration
        min_music_for_ad = ad_end + fade_duration + 1.0
        if music_duration < min_music_for_ad:
            log.warning(f"  歌曲过短 ({music_duration:.1f}s)，调整结尾")
            best_end_time = min(best_end_time, music_duration - fade_duration)

        effective_target = best_end_time + fade_duration

        # ④ 混音
        ok = mix_with_ffmpeg(
            str(music_path), str(ad_path), str(output_path),
            effective_target, ad_start, ad_duration,
            duck_volume, fade_duration, ad_volume,
        )

        if not ok:
            raise HTTPException(500, "混音失败，请检查音频文件")

        log.info(f"请求 {session_id} 完成: {output_path}")

        # 返回文件后自动清理
        return FileResponse(
            path=str(output_path),
            media_type="audio/mpeg",
            filename=f"mixed_{session_id}.mp3",
            background=lambda: shutil.rmtree(work_dir, ignore_errors=True),
        )

    except HTTPException:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(work_dir, ignore_errors=True)
        log.exception(f"请求 {session_id} 异常")
        raise HTTPException(500, f"处理失败: {str(e)}")


@app.post("/analyze")
async def analyze_only(
    music: UploadFile = File(..., description="歌曲音频文件"),
) -> JSONResponse:
    """仅分析音频，返回时长和能量特征（用于调试/预览）"""
    music_ext = Path(music.filename or "music.mp3").suffix.lower()
    session_id = uuid.uuid4().hex[:8]
    work_dir = TEMP_DIR / session_id
    work_dir.mkdir(parents=True, exist_ok=True)
    music_path = work_dir / f"music_{session_id}{music_ext}"

    try:
        with open(music_path, "wb") as f:
            f.write(await music.read())

        duration = get_duration_ffprobe(str(music_path))
        analysis = analyze_audio(str(music_path), sr=22050)

        rms_samples = []
        for i in range(0, len(analysis["rms_smooth"]), max(1, len(analysis["rms_smooth"]) // 100)):
            rms_samples.append({
                "time": round(float(analysis["rms_times"][i]), 2),
                "rms": round(float(analysis["rms_smooth"][i]), 6),
            })

        return JSONResponse({
            "duration": round(duration, 2),
            "sample_rate": analysis["sr"],
            "frames": len(analysis["rms_smooth"]),
            "rms_samples": rms_samples,
        })

    except Exception as e:
        log.exception("分析失败")
        raise HTTPException(500, f"分析失败: {str(e)}")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
