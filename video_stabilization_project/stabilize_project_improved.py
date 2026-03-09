import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


INPUT_VIDEO = "input/pexels_4101533.mp4"
OUTPUT_DIR = "output_improved_v2"

OUTPUT_STABILIZED = os.path.join(OUTPUT_DIR, "stabilized_improved_v2.mp4")
OUTPUT_COMPARISON = os.path.join(OUTPUT_DIR, "comparison_improved_v2.mp4")
OUTPUT_TRAJECTORY = os.path.join(OUTPUT_DIR, "trajectory_improved_v2.png")
OUTPUT_METRICS = os.path.join(OUTPUT_DIR, "metrics_improved_v2.txt")
OUTPUT_SAMPLE_1 = os.path.join(OUTPUT_DIR, "sample_01.jpg")
OUTPUT_SAMPLE_2 = os.path.join(OUTPUT_DIR, "sample_02.jpg")


SMOOTHING_RADIUS = 12
EMA_ALPHA = 0.30
BORDER_SCALE = 1.13

MAX_CORNERS = 700
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 10
BLOCK_SIZE = 3

LK_PARAMS = dict(
    winSize=(25, 25),
    maxLevel=4,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

RANSAC_REPROJ_THRESHOLD = 1.8
MAX_FLOW_MAGNITUDE = 55.0
MIN_VALID_POINTS = 20
FB_ERROR_THRESHOLD = 1.5

SIDE_MARGIN_RATIO = 0.25
TOP_MARGIN_RATIO = 0.08
BOTTOM_MARGIN_RATIO = 0.92

SAMPLE_RATIOS = [0.30, 0.75]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def moving_average(curve: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return curve.copy()
    window_size = 2 * radius + 1
    filt = np.ones(window_size, dtype=np.float32) / window_size
    padded = np.pad(curve, (radius, radius), mode="edge")
    smoothed = np.convolve(padded, filt, mode="same")
    return smoothed[radius:-radius]


def smooth_trajectory_ma(trajectory: np.ndarray, radius: int) -> np.ndarray:
    out = trajectory.copy()
    for i in range(trajectory.shape[1]):
        out[:, i] = moving_average(trajectory[:, i], radius)
    return out


def smooth_trajectory_ema(trajectory: np.ndarray, alpha: float) -> np.ndarray:
    out = trajectory.copy()
    for j in range(trajectory.shape[1]):
        for i in range(1, len(trajectory)):
            out[i, j] = alpha * trajectory[i, j] + (1.0 - alpha) * out[i - 1, j]
    return out


def combined_smoothing(trajectory: np.ndarray) -> np.ndarray:
    ema = smooth_trajectory_ema(trajectory, EMA_ALPHA)
    ma = smooth_trajectory_ma(ema, SMOOTHING_RADIUS)
    return ma


def build_side_feature_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)

    y1 = int(height * TOP_MARGIN_RATIO)
    y2 = int(height * BOTTOM_MARGIN_RATIO)

    left_x2 = int(width * SIDE_MARGIN_RATIO)
    right_x1 = int(width * (1.0 - SIDE_MARGIN_RATIO))

    mask[y1:y2, 0:left_x2] = 255
    mask[y1:y2, right_x1:width] = 255

    top_band_y2 = int(height * 0.18)
    center_x1 = int(width * 0.44)
    center_x2 = int(width * 0.56)
    mask[0:top_band_y2, center_x1:center_x2] = 255

    return mask


def fix_border(frame: np.ndarray, scale: float = 1.13) -> np.ndarray:
    h, w = frame.shape[:2]
    T = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
    return cv2.warpAffine(
        frame,
        T,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )


def affine_from_params(dx: float, dy: float, da: float) -> np.ndarray:
    c = np.cos(da)
    s = np.sin(da)
    return np.array([
        [c, -s, dx],
        [s,  c, dy]
    ], dtype=np.float32)


def black_ratio(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray < 5))


def save_side_by_side(before: np.ndarray, after: np.ndarray, out_path: str) -> None:
    both = np.hstack([before, after])
    cv2.imwrite(out_path, both)


def plot_trajectory(original: np.ndarray, smoothed: np.ndarray, out_path: str) -> None:
    labels = ["Accumulated dx", "Accumulated dy", "Accumulated angle"]
    plt.figure(figsize=(12, 8))
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(original[:, i], label="Original")
        plt.plot(smoothed[:, i], label="Smoothed")
        plt.ylabel(labels[i])
        plt.grid(True)
        plt.legend()
    plt.xlabel("Frame index")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def preprocess_gray(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def estimate_transform(prev_gray: np.ndarray, curr_gray: np.ndarray, feature_mask: np.ndarray):
    prev_gray_eq = preprocess_gray(prev_gray)
    curr_gray_eq = preprocess_gray(curr_gray)

    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray_eq,
        maxCorners=MAX_CORNERS,
        qualityLevel=QUALITY_LEVEL,
        minDistance=MIN_DISTANCE,
        blockSize=BLOCK_SIZE,
        mask=feature_mask
    )

    identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    if prev_pts is None or len(prev_pts) < MIN_VALID_POINTS:
        return 0.0, 0.0, 0.0, identity, 0, 0

    curr_pts, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray_eq, curr_gray_eq, prev_pts, None, **LK_PARAMS
    )

    if curr_pts is None:
        return 0.0, 0.0, 0.0, identity, 0, 0

    back_pts, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
        curr_gray_eq, prev_gray_eq, curr_pts, None, **LK_PARAMS
    )

    if back_pts is None:
        return 0.0, 0.0, 0.0, identity, 0, 0

    status_fwd = status_fwd.reshape(-1).astype(bool)
    status_bwd = status_bwd.reshape(-1).astype(bool)
    status = status_fwd & status_bwd

    prev_pts = prev_pts[status]
    curr_pts = curr_pts[status]
    back_pts = back_pts[status]

    if len(prev_pts) < MIN_VALID_POINTS:
        return 0.0, 0.0, 0.0, identity, len(prev_pts), 0

    # Forward-backward consistency
    fb_error = np.linalg.norm((prev_pts - back_pts).reshape(-1, 2), axis=1)
    keep_fb = fb_error < FB_ERROR_THRESHOLD

    prev_pts = prev_pts[keep_fb]
    curr_pts = curr_pts[keep_fb]

    if len(prev_pts) < MIN_VALID_POINTS:
        return 0.0, 0.0, 0.0, identity, len(prev_pts), 0

    # Фильтрация по величине потока
    flow = (curr_pts - prev_pts).reshape(-1, 2)
    flow_mag = np.linalg.norm(flow, axis=1)
    keep_mag = flow_mag < MAX_FLOW_MAGNITUDE

    prev_pts = prev_pts[keep_mag]
    curr_pts = curr_pts[keep_mag]

    if len(prev_pts) < MIN_VALID_POINTS:
        return 0.0, 0.0, 0.0, identity, len(prev_pts), 0

    # Робастная фильтрация по медиане смещения
    flow = (curr_pts - prev_pts).reshape(-1, 2)
    median_flow = np.median(flow, axis=0)
    deviation = np.linalg.norm(flow - median_flow, axis=1)

    mad = np.median(np.abs(deviation - np.median(deviation))) + 1e-6
    keep_med = deviation < (2.5 * mad + 2.0)

    prev_pts = prev_pts[keep_med]
    curr_pts = curr_pts[keep_med]

    if len(prev_pts) < MIN_VALID_POINTS:
        return 0.0, 0.0, 0.0, identity, len(prev_pts), 0

    m, inliers = cv2.estimateAffinePartial2D(
        prev_pts,
        curr_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD,
        maxIters=4000,
        confidence=0.995,
        refineIters=30
    )

    if m is None:
        return 0.0, 0.0, 0.0, identity, len(prev_pts), 0

    inliers_count = int(inliers.sum()) if inliers is not None else 0

    dx = float(m[0, 2])
    dy = float(m[1, 2])
    da = float(np.arctan2(m[1, 0], m[0, 0]))

    return dx, dy, da, m.astype(np.float32), len(prev_pts), inliers_count


def estimate_all_transforms(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    ok, prev = cap.read()
    if not ok:
        raise RuntimeError("Cannot read first frame.")

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    feature_mask = build_side_feature_mask(height, width)

    transforms = np.zeros((n_frames - 1, 3), dtype=np.float32)
    stats = []

    print(f"[INFO] Frames={n_frames}, size={width}x{height}, fps={fps:.2f}")

    for i in range(n_frames - 1):
        ok, curr = cap.read()
        if not ok:
            transforms = transforms[:i]
            break

        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        dx, dy, da, _, valid_pts, inliers = estimate_transform(prev_gray, curr_gray, feature_mask)
        transforms[i] = [dx, dy, da]
        stats.append((valid_pts, inliers))

        prev_gray = curr_gray

        if i % 20 == 0:
            print(f"[INFO] Estimated transforms: {i + 1}/{n_frames - 1}")

    cap.release()
    return transforms, stats, width, height, fps


def stabilize_video(video_path: str, transforms: np.ndarray, transforms_smooth: np.ndarray,
                    width: int, height: int, fps: float):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_stab = cv2.VideoWriter(OUTPUT_STABILIZED, fourcc, fps, (width, height))
    out_comp = cv2.VideoWriter(OUTPUT_COMPARISON, fourcc, fps, (width * 2, height))

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Cannot read first frame for stabilization.")

    first = fix_border(frame, BORDER_SCALE)
    before = frame.copy()
    after = first.copy()

    cv2.putText(before, "Before", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(after, "After", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    out_stab.write(first)
    out_comp.write(np.hstack([before, after]))

    black_vals = [black_ratio(first)]

    sample_ids = [int(len(transforms_smooth) * r) for r in SAMPLE_RATIOS]
    sample_saved = 0

    for i in range(len(transforms_smooth)):
        ok, frame = cap.read()
        if not ok:
            break

        # Важно: применяем сглаженное преобразование напрямую
        dx = transforms_smooth[i, 0]
        dy = transforms_smooth[i, 1]
        da = transforms_smooth[i, 2]

        m = affine_from_params(dx, dy, da)

        stabilized = cv2.warpAffine(
            frame,
            m,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        stabilized = fix_border(stabilized, BORDER_SCALE)

        before = frame.copy()
        after = stabilized.copy()

        cv2.putText(before, "Before", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(after, "After", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        out_stab.write(stabilized)
        out_comp.write(np.hstack([before, after]))

        if i in sample_ids:
            if sample_saved == 0:
                save_side_by_side(frame, stabilized, OUTPUT_SAMPLE_1)
                sample_saved += 1
            elif sample_saved == 1:
                save_side_by_side(frame, stabilized, OUTPUT_SAMPLE_2)
                sample_saved += 1

        black_vals.append(black_ratio(stabilized))

        if i % 20 == 0:
            print(f"[INFO] Stabilized frames: {i + 1}/{len(transforms_smooth)}")

    cap.release()
    out_stab.release()
    out_comp.release()

    return float(np.mean(black_vals))


def compute_metrics(raw_transforms: np.ndarray, smooth_transforms: np.ndarray, stats):
    raw_motion = np.linalg.norm(raw_transforms[:, :2], axis=1)
    smooth_motion = np.linalg.norm(smooth_transforms[:, :2], axis=1)

    valid_pts_mean = float(np.mean([s[0] for s in stats])) if stats else 0.0
    inliers_mean = float(np.mean([s[1] for s in stats])) if stats else 0.0

    return {
        "valid_pts_mean": valid_pts_mean,
        "inliers_mean": inliers_mean,
        "raw_mean_translation": float(np.mean(raw_motion)),
        "smooth_mean_translation": float(np.mean(smooth_motion)),
        "raw_std_translation": float(np.std(raw_motion)),
        "smooth_std_translation": float(np.std(smooth_motion)),
        "raw_mean_abs_angle": float(np.mean(np.abs(raw_transforms[:, 2]))),
        "smooth_mean_abs_angle": float(np.mean(np.abs(smooth_transforms[:, 2]))),
    }


def main():
    ensure_dir(OUTPUT_DIR)

    transforms, stats, width, height, fps = estimate_all_transforms(INPUT_VIDEO)

    if len(transforms) < 5:
        raise RuntimeError("Too few frames/transforms.")

    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = combined_smoothing(trajectory)

    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference

    plot_trajectory(trajectory, smoothed_trajectory, OUTPUT_TRAJECTORY)

    mean_black = stabilize_video(
        INPUT_VIDEO,
        transforms,
        transforms_smooth,
        width,
        height,
        fps
    )

    metrics = compute_metrics(transforms, transforms_smooth, stats)

    with open(OUTPUT_METRICS, "w", encoding="utf-8") as f:
        f.write("=== Improved Video Stabilization Metrics v2 ===\n")
        f.write(f"Input video: {INPUT_VIDEO}\n")
        f.write(f"Mean tracked points: {metrics['valid_pts_mean']:.2f}\n")
        f.write(f"Mean RANSAC inliers: {metrics['inliers_mean']:.2f}\n")
        f.write(f"Mean black ratio: {mean_black:.6f}\n")
        f.write(f"Raw mean translation: {metrics['raw_mean_translation']:.6f}\n")
        f.write(f"Smoothed mean translation: {metrics['smooth_mean_translation']:.6f}\n")
        f.write(f"Raw std translation: {metrics['raw_std_translation']:.6f}\n")
        f.write(f"Smoothed std translation: {metrics['smooth_std_translation']:.6f}\n")
        f.write(f"Raw mean abs angle: {metrics['raw_mean_abs_angle']:.6f}\n")
        f.write(f"Smoothed mean abs angle: {metrics['smooth_mean_abs_angle']:.6f}\n")

    print("[DONE]")
    print(f"[OUT] {OUTPUT_STABILIZED}")
    print(f"[OUT] {OUTPUT_COMPARISON}")
    print(f"[OUT] {OUTPUT_TRAJECTORY}")
    print(f"[OUT] {OUTPUT_METRICS}")
    print(f"[OUT] {OUTPUT_SAMPLE_1}")
    print(f"[OUT] {OUTPUT_SAMPLE_2}")


if __name__ == "__main__":
    main()