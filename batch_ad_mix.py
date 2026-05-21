#!/usr/bin/env python3
"""
批量广告混音脚本 - batch_ad_mix.py
功能：自动分析歌曲特征 → 找最佳广告插入点 → 混音+淡出+裁剪
依赖：pip install imageio-ffmpeg numpy
用法：
    python batch_ad_mix.py --music_dir input_music --ad ad/ad.mp3 --out_dir output
"""

import argparse
import sys
import json
import logging
import subprocess
import numpy as np
from pathlib import Path

# ── FFmpeg 路径（imageio-ffmpeg 自带）─────────────────────────────
import imageio_ffmpeg
FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()


# ── 日志配置 ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("batch_ad_mix")


# ═══════════════════════════════════════════════════════════════════
#  核心算法
# ═══════════════════════════════════════════════════════════════════

def analyze_audio(audio_path: str, sr: int = 22050):
    """
    用 FFmpeg 解码为 mono float32 后分析音频，返回：
      - duration: 总时长（秒）
      - rms_energy: 每帧 RMS 能量数组（跟 music_length 对应）
      - rms_times:  每帧起始时间数组
      - rms_smooth: 能量平滑值（用于找低能量段）
    """
    log.info(f"分析音频: {audio_path}")
    cmd = [
        FFMPEG_BIN,
        "-v", "error",
        "-i", audio_path,
        "-vn",
        "-ac", "1",
        "-ar", str(sr),
        "-f", "f32le",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"FFmpeg 解码失败: {stderr[-500:]}")

    y = np.frombuffer(result.stdout, dtype=np.float32)
    duration = len(y) / sr

    # 帧长 2048，hop 512 → 约 43ms/帧
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

    # 平滑能量（移动均值，窗口约 1 秒）
    window_size = max(1, int(sr / hop_length * 1.0))
    rms_smooth = np.convolve(rms, np.ones(window_size) / window_size, mode="same")

    log.info(f"  总时长 {duration:.1f}s，帧数 {len(rms)}，采样率 {sr}")
    return {
        "duration": float(duration),
        "rms": rms,
        "rms_times": times,
        "rms_smooth": rms_smooth,
        "sr": sr,
        "hop": hop_length,
    }


def find_best_insertion_point(analysis: dict, search_start_ratio: float,
                              search_end_ratio: float, ad_duration: float,
                              target_duration: float | None = None,
                              fade_duration: float = 0.0,
                              search_start_sec: float | None = None,
                              search_end_sec: float | None = None) -> float:
    """
    在歌曲 search_start_ratio%~search_end_ratio% 之间找低能量、变化平稳的位置。
    返回最佳插入点（秒）。
    """
    rms = analysis["rms_smooth"]
    times = analysis["rms_times"]
    total = analysis["duration"]

    if search_start_sec is not None and search_end_sec is not None:
        start_time = search_start_sec
        end_time = search_end_sec
    else:
        start_time = total * search_start_ratio
        end_time = total * search_end_ratio

    # 过滤出搜索范围
    mask = (times >= start_time) & (times <= end_time)
    if not mask.any():
        log.warning(f"  搜索范围为空，使用中间点")
        return total * 0.5

    segment_rms = rms[mask]
    segment_times = times[mask]

    # 归一化能量 (0~1)
    rms_min, rms_max = segment_rms.min(), segment_rms.max()
    if rms_max > rms_min:
        rms_norm = (segment_rms - rms_min) / (rms_max - rms_min)
    else:
        rms_norm = np.zeros_like(segment_rms)

    # ── 策略：低能量且变化平稳 ────────────────────────────────
    # (1) 能量分（越低越好）
    energy_score = 1.0 - rms_norm

    # (2) 平稳分（相邻帧变化越小越好）
    diff = np.abs(np.diff(segment_rms))
    diff_pad = np.concatenate([[0], diff])
    stability_score = 1.0 - diff_pad / (diff_pad.max() + 1e-9)

    # (3) 综合分（能量 60% + 平稳 40%）
    combined = 0.6 * energy_score + 0.4 * stability_score

    # (4) 避开广告结束前 5s（广告结束后要留足音乐）
    margin = 5.0
    available_end = total - ad_duration - margin
    if target_duration is not None:
        available_end = min(available_end, target_duration - ad_duration - fade_duration - 1.0)
    avail_mask = segment_times <= available_end
    if not avail_mask.any():
        fallback_time = max(0.0, available_end)
        log.warning(f"  搜索范围内没有可容纳广告的位置，使用 {fallback_time:.2f}s")
        return fallback_time

    combined_masked = np.where(avail_mask, combined, -np.inf)
    best_idx = int(np.argmax(combined_masked))
    best_time = float(segment_times[best_idx])

    log.info(f"  搜索范围: {start_time:.1f}s ~ {end_time:.1f}s")
    log.info(f"  能量范围: {rms_min:.4f} ~ {rms_max:.4f}")
    log.info(f"  最佳插入点: {best_time:.2f}s（综合得分 {combined[best_idx]:.3f}）")

    return best_time


def find_best_end_point(analysis: dict, target_duration: float,
                       ad_start: float, ad_duration: float,
                       fade_duration: float, search_margin: float = 10.0,
                       max_duration_ratio: float = 0.6) -> float:
    """
    在目标时长附近找最佳结尾点。
    策略：找能量下降点、乐句结束位置、避免在高潮中间截断。
    
    返回：最佳结尾时间点（秒）
    """
    rms = analysis["rms_smooth"]
    times = analysis["rms_times"]
    total = analysis["duration"]
    
    # ── 确定搜索范围 ──────────────────────────────────────────
    # 广告结束后至少要留 fade_duration + 1s 的音乐
    ad_end = ad_start + ad_duration
    min_music_after_ad = fade_duration + 1.0
    min_end = ad_end + min_music_after_ad
    max_end = max(min_end, total * max_duration_ratio - fade_duration)
    
    # 搜索范围：广告结束后开始，target 只作为参考上限，不强行贴近。
    search_start = max(min_end, ad_end + 15.0)
    search_end = min(total - fade_duration, max_end, max(target_duration + search_margin, search_start + 20.0))
    
    if search_end <= search_start:
        log.warning(f"  结尾搜索范围不足，使用最小值: {min_end:.1f}s")
        return min(min_end, total - fade_duration)
    
    # ── 分析结尾候选点 ──────────────────────────────────────────
    mask = (times >= search_start) & (times <= search_end)
    if not mask.any():
        log.warning(f"  结尾候选段为空")
        return min(target_duration, total - fade_duration)
    
    segment_rms = rms[mask]
    segment_times = times[mask]
    
    # ── 策略：找能量下降点、低能量平稳区、局部谷底 ────────────────
    # (1) 能量下降趋势：找能量从高到低的过渡点
    # 计算向后差分（负值表示下降）
    backward_diff = np.diff(np.concatenate([segment_rms, [segment_rms[-1]]]))[::-1]
    backward_diff_forward = -backward_diff[::-1]  # 正值表示下降趋势
    
    # (2) 能量水平：优先低能量区域
    rms_min, rms_max = segment_rms.min(), segment_rms.max()
    if rms_max > rms_min:
        rms_norm = (segment_rms - rms_min) / (rms_max - rms_min)
    else:
        rms_norm = np.zeros_like(segment_rms)
    low_energy_score = 1.0 - rms_norm  # 低能量更好
    
    # (3) 变化平稳：找能量变化小的位置（乐句结束通常稳定）
    forward_diff = np.abs(np.diff(np.concatenate([[segment_rms[0]], segment_rms])))
    stability_score = 1.0 - forward_diff / (forward_diff.max() + 1e-9)

    # (4) 局部谷底：前后约 2 秒内能量更低的位置，通常更像段落收束点
    local_window = max(3, int(2.0 / (analysis["hop"] / analysis["sr"])))
    valley_score = np.zeros_like(segment_rms)
    for i in range(len(segment_rms)):
        left = max(0, i - local_window)
        right = min(len(segment_rms), i + local_window + 1)
        local_min = segment_rms[left:right].min()
        local_max = segment_rms[left:right].max()
        if local_max > local_min:
            valley_score[i] = 1.0 - (segment_rms[i] - local_min) / (local_max - local_min)
    
    # ── 综合评分 ────────────────────────────────────────────────
    # 权重：优先音乐段落自然收束，其次才考虑长度
    decline_score = backward_diff_forward / (backward_diff_forward.max() + 1e-9)
    combined = 0.35 * low_energy_score + 0.25 * stability_score + 0.25 * decline_score + 0.15 * valley_score
    
    # 轻微倾向参考时长，避免过长；但不压过音乐节奏判断。
    over_target_penalty = np.maximum(segment_times - target_duration, 0) / max(search_margin, 1.0)
    combined = combined - 0.05 * over_target_penalty
    
    # ── 找最佳点 ────────────────────────────────────────────────
    best_idx = int(np.argmax(combined))
    best_time = float(segment_times[best_idx])
    
    # 确保至少留 fade_duration 做淡出
    final_end = min(best_time, total - fade_duration, max_end)
    
    log.info(f"  结尾搜索: {search_start:.1f}s ~ {search_end:.1f}s (歌曲总长 {total:.1f}s)")
    log.info(f"  能量范围: {rms_min:.4f} ~ {rms_max:.4f}")
    log.info(f"  最佳结尾点: {final_end:.2f}s（淡出 {fade_duration}s → 结束于 {final_end+fade_duration:.2f}s）")
    
    return final_end


def build_ffmpeg_filter(music_duration: float, ad_duration: float,
                        ad_start: float, target_duration: float,
                        duck_volume: float, fade_duration: float,
                        ad_volume: float) -> tuple:
    """
    构建 ffmpeg filter_complex 字符串。

    广告前：正常音量
    广告前淡出：逐渐降到 duck_volume
    广告中：duck_volume
    广告后淡入：逐渐恢复正常
    广告后：正常音量
    结尾：淡出

    返回 (filter_complex, input_labels)
    """
    ad_end = ad_start + ad_duration
    fade_start = target_duration - fade_duration

    # ── 计算音量淡入淡出时间点 ──────────────────────────────
    # 广告开始前 0.5s 开始降音
    duck_start = max(ad_start - 0.5, 0.0)
    # 广告结束后 0.5s 恢复正常
    duck_end = min(ad_end + 0.5, target_duration)

    # 音量表达式（使用 if 条件）
    # between(t,duck_start,ad_start)：广告前淡出
    # between(t,ad_start,ad_end)：广告中保持低
    # between(t,ad_end,duck_end)：广告后淡入
    # 其余时间正常
    music_vol = (
        f"if(between(t,{duck_start:.3f},{ad_start:.3f}),"
        f"{duck_volume:.2f}+(1-{duck_volume:.2f})*(1-(t-{duck_start:.3f})/{max(ad_start-duck_start,0.001):.3f}),"
        f"if(between(t,{ad_start:.3f},{ad_end:.3f}),"
        f"{duck_volume:.2f},"
        f"if(between(t,{ad_end:.3f},{duck_end:.3f}),"
        f"{duck_volume:.2f}+(1-{duck_volume:.2f})*(t-{ad_end:.3f})/{max(duck_end-ad_end,0.001):.3f},"
        f"1)))"
    )

    # 结尾淡出（全局最后 fade_duration 秒）
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
    """
    调用 ffmpeg 执行混音。
    """
    filter_complex, output_label = build_ffmpeg_filter(
        music_duration=target_duration,  # 只用到 target_duration 做 atrim
        ad_duration=ad_duration,
        ad_start=ad_start,
        target_duration=target_duration,
        duck_volume=duck_volume,
        fade_duration=fade_duration,
        ad_volume=ad_volume,
    )

    cmd = [
        FFMPEG_BIN,
        "-y",
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

    log.info(f"  FFmpeg 命令: {' '.join(cmd[:6])} ...")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
        if result.returncode == 0:
            log.info(f"  OK 成功: {output_path}")
            return True
        else:
            log.error(f"  ERR FFmpeg 失败: {result.stderr[-500:]}")
            return False
    except subprocess.TimeoutExpired:
        log.error("  ERR FFmpeg 超时")
        return False


def get_duration_ffprobe(audio_path: str) -> float:
    """用 ffprobe 精确获取音频时长（秒）"""
    cmd = [
        FFMPEG_BIN.replace("ffmpeg", "ffprobe"),
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        audio_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except Exception:
        analysis = analyze_audio(audio_path)
        return analysis["duration"]


# ═══════════════════════════════════════════════════════════════════
#  批量处理主流程
# ═══════════════════════════════════════════════════════════════════

def process_single(music_path: str, ad_path: str, output_dir: Path, args) -> dict:
    """处理单首歌曲，返回日志记录 dict"""
    music_name = Path(music_path).stem
    ad_name = Path(ad_path).stem
    log.info("=" * 60)
    log.info(f"处理: {music_name} + 广告 {ad_name}")

    record = {
        "music": Path(music_path).name,
        "ad": Path(ad_path).name,
        "original_duration_s": 0,
        "target_duration_s": args.target_duration,
        "actual_duration_s": 0,
        "ad_duration_s": 0,
        "ad_start_s": 0,
        "best_end_point_s": 0,
        "fade_range_s": "",
        "output_file": "",
        "status": "pending",
    }

    # ① 获取时长
    try:
        music_duration = get_duration_ffprobe(music_path)
        ad_duration = get_duration_ffprobe(ad_path)
    except Exception as e:
        log.error(f"  ✗ 无法读取音频: {e}")
        record["status"] = "error_read"
        return record

    record["original_duration_s"] = round(music_duration, 1)
    record["ad_duration_s"] = round(ad_duration, 2)

    log.info(f"  歌曲时长: {music_duration:.1f}s，广告时长: {ad_duration:.1f}s")

    # ② 分析歌曲找插入点
    try:
        analysis = analyze_audio(music_path)
    except Exception as e:
        log.error(f"  ✗ 音频分析失败: {e}")
        record["status"] = "error_analyze"
        return record

    ad_start = find_best_insertion_point(
        analysis,
        search_start_ratio=args.search_start_ratio,
        search_end_ratio=args.search_end_ratio,
        ad_duration=ad_duration,
        target_duration=args.target_duration,
        fade_duration=args.fade_duration,
        search_start_sec=args.search_start_sec,
        search_end_sec=args.search_end_sec,
    )

    # ③ 找最佳结尾点（智能裁剪，不硬切）
    best_end_time = find_best_end_point(
        analysis,
        target_duration=args.target_duration,
        ad_start=ad_start,
        ad_duration=ad_duration,
        fade_duration=args.fade_duration,
        search_margin=args.end_search_margin,
        max_duration_ratio=args.max_duration_ratio,
    )

    # 实际成品时长 = 最佳结尾点 + 淡出时长
    # （find_best_end_point 已考虑淡出空间）
    ad_end = ad_start + ad_duration
    min_music_for_ad = ad_end + args.fade_duration + 1.0
    
    if music_duration < min_music_for_ad:
        log.warning(f"  歌曲过短（{music_duration:.1f}s），调整结尾")
        best_end_time = min(best_end_time, music_duration - args.fade_duration)
    
    # 最终输出时长（含淡出）
    effective_target = best_end_time + args.fade_duration
    
    # ④ 输出文件名
    safe_name = music_name.replace("/", "_").replace("\\", "_")
    safe_ad_name = ad_name.replace("/", "_").replace("\\", "_")
    out_name = f"{safe_name}_{safe_ad_name}_广告版_{int(effective_target)}s.mp3"
    output_path = output_dir / out_name
    record["output_file"] = out_name
    record["ad_start_s"] = round(ad_start, 2)
    record["best_end_point_s"] = round(best_end_time, 2)
    record["actual_duration_s"] = round(effective_target, 1)

    # ⑤ 调用 ffmpeg 混音
    ok = mix_with_ffmpeg(
        music_path=music_path,
        ad_path=ad_path,
        output_path=str(output_path),
        target_duration=effective_target,
        ad_start=ad_start,
        ad_duration=ad_duration,
        duck_volume=args.duck_volume,
        fade_duration=args.fade_duration,
        ad_volume=args.ad_volume,
    )

    record["status"] = "success" if ok else "error_ffmpeg"

    fade_end = effective_target
    fade_start = effective_target - args.fade_duration
    record["fade_range_s"] = f"{fade_start:.1f}~{fade_end:.1f}s"

    if ok:
        record["output_size_kb"] = round(output_path.stat().st_size / 1024, 1)
    else:
        record["output_size_kb"] = 0

    return record


def batch_process(args):
    """批量处理入口"""
    music_dir = Path(args.music_dir).resolve()
    ad_dir = Path(args.ad_dir).resolve()
    output_dir = Path(args.out_dir).resolve()
    suffixes = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}

    if not music_dir.exists():
        log.error(f"音乐目录不存在: {music_dir}")
        sys.exit(1)
    if args.ad:
        ad_files = [Path(args.ad).resolve()]
        missing_ads = [p for p in ad_files if not p.exists()]
        if missing_ads:
            log.error(f"广告文件不存在: {missing_ads[0]}")
            sys.exit(1)
    else:
        if not ad_dir.exists():
            log.error(f"广告目录不存在: {ad_dir}")
            sys.exit(1)
        ad_files = [f for f in ad_dir.iterdir() if f.suffix.lower() in suffixes]

    if not ad_files:
        log.error(f"未找到广告音频（支持: {suffixes}）")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    music_files = [f for f in music_dir.iterdir() if f.suffix.lower() in suffixes]

    if not music_files:
        log.error(f"目录 {music_dir} 中未找到音频文件（支持: {suffixes}）")
        sys.exit(1)

    log.info(f"找到 {len(music_files)} 首歌曲")
    log.info(f"找到 {len(ad_files)} 条广告")
    log.info(f"输出目录: {output_dir}")
    log.info(f"目标时长: {args.target_duration}s | 背景音乐音量: {args.duck_volume} | 广告音量: {args.ad_volume} | 淡出: {args.fade_duration}s")
    if args.search_start_sec is not None and args.search_end_sec is not None:
        log.info(f"广告插入搜索: {args.search_start_sec:.0f}s ~ {args.search_end_sec:.0f}s | 结尾搜索: ±{args.end_search_margin}s")
    else:
        log.info(f"广告插入搜索: {args.search_start_ratio*100:.0f}% ~ {args.search_end_ratio*100:.0f}% | 结尾搜索: ±{args.end_search_margin}s")
    log.info("-" * 60)

    results = []
    for ad_file in ad_files:
        for music_file in music_files:
            record = process_single(str(music_file), str(ad_file), output_dir, args)
            results.append(record)

    # ── 生成日志报告 ───────────────────────────────────────────
    log.info("=" * 60)
    log.info("处理完成！")

    report_path = output_dir / "batch_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    success_n = sum(1 for r in results if r["status"] == "success")
    fail_n = len(results) - success_n

    # 控制台表格
    print("\n" + "═" * 90)
    print(f"{'歌曲':<18} {'广告':<18} {'状态':<8} {'原时长':>6} {'实际':>5} {'广告起点':>7} {'结尾点':>6} {'输出文件':<25}")
    print("─" * 90)
    for r in results:
        status_icon = "OK" if r["status"] == "success" else "ERR"
        print(
            f"{r['music'][:17]:<18} {r.get('ad','')[:17]:<18} {status_icon}{r['status'][:6]:<6}"
            f" {r['original_duration_s']:>5.1f}s"
            f" {r.get('actual_duration_s',r['target_duration_s']):>4.1f}s"
            f" {r['ad_start_s']:>6.2f}s"
            f" {r.get('best_end_point_s',0):>5.1f}s"
            f" {r['output_file'][:24]:<25}"
        )
    print("═" * 90)
    print(f"成功: {success_n}/{len(results)}，失败: {fail_n}，详细日志: {report_path}")
    print(f"输出目录: {output_dir}")

    # 打印每条失败记录原因
    if fail_n > 0:
        print("\n失败详情:")
        for r in results:
            if r["status"] != "success":
                print(f"  - {r['music']} + {r.get('ad','')}: {r['status']}")

    return results


# ═══════════════════════════════════════════════════════════════════
#  参数解析 & 入口
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="批量广告混音脚本：自动找点 + ffmpeg 混音 + 批量输出",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python batch_ad_mix.py
  python batch_ad_mix.py --music_dir input --ad_dir ad --out_dir output
  python batch_ad_mix.py --ad ad/ad.mp3 --duck_volume 0.3 --ad_volume 2.3 --target_duration 120
        """,
    )
    parser.add_argument("--music_dir", default="input", help="歌曲文件夹路径（默认: input）")
    parser.add_argument("--ad_dir", default="ad", help="广告音频文件夹路径（默认: ad）")
    parser.add_argument("--ad", default=None, help="只处理单个广告音频文件；不传则处理 ad_dir 中全部广告")
    parser.add_argument("--out_dir", default="output", help="输出目录（默认: output）")
    parser.add_argument("--target_duration", type=int, default=120,
                        help="目标成品时长（秒，默认: 120）")
    parser.add_argument("--duck_volume", type=float, default=0.3,
                        help="广告期间背景音乐音量（默认: 0.3，即 30%%）")
    parser.add_argument("--ad_volume", type=float, default=2.3,
                        help="广告音频音量倍数（默认: 2.3，即 230%%）")
    parser.add_argument("--fade_duration", type=float, default=3.0,
                        help="结尾淡出时长（秒，默认: 3）")
    parser.add_argument("--search_start_ratio", type=float, default=0.35,
                        help="广告插入点搜索起点（占歌曲比例；当 search_start_sec/search_end_sec 未设置时使用）")
    parser.add_argument("--search_end_ratio", type=float, default=0.70,
                        help="广告插入点搜索终点（占歌曲比例；当 search_start_sec/search_end_sec 未设置时使用）")
    parser.add_argument("--search_start_sec", type=float, default=20.0,
                        help="广告插入点搜索起点秒数（默认: 20）")
    parser.add_argument("--search_end_sec", type=float, default=40.0,
                        help="广告插入点搜索终点秒数（默认: 40）")
    parser.add_argument("--end_search_margin", type=float, default=60.0,
                        help="结尾裁剪搜索范围上限扩展秒数（默认: 60；值越大越按音乐节奏找结尾）")
    parser.add_argument("--max_duration_ratio", type=float, default=0.6,
                        help="成品最长不超过原歌曲的比例（默认: 0.6，即 60%%）")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="日志级别（默认: INFO）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.getLogger("batch_ad_mix").setLevel(getattr(logging, args.log_level))
    batch_process(args)
