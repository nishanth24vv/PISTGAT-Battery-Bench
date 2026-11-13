"""Generate crisp 15-second split-screen demonstration hero GIF for README."""

import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_demo_gif(output_path="docs/demo.gif", num_frames=45, duration_per_frame_ms=330):
    # Total duration = 45 frames * 330ms = ~15 seconds
    width, height = 860, 450
    frames = []

    # Try loading default system fonts or fallback
    try:
        font_mono_small = ImageFont.truetype("consola.ttf", 11)
        font_mono = ImageFont.truetype("consola.ttf", 13)
        font_mono_bold = ImageFont.truetype("consolab.ttf", 14)
        font_title = ImageFont.truetype("arialbd.ttf", 16)
        font_sub = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font_mono_small = ImageFont.load_default()
        font_mono = ImageFont.load_default()
        font_mono_bold = ImageFont.load_default()
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    for f_idx in range(num_frames):
        img = Image.new("RGB", (width, height), (10, 15, 26))
        draw = ImageDraw.Draw(img)

        progress = f_idx / (num_frames - 1)
        cycle_current = int(1 + progress * 59)

        # -------------------------------------------------------------
        # Left Pane: CLI Terminal Run (x: 15 to 415, y: 15 to 435)
        # -------------------------------------------------------------
        draw.rounded_rectangle([15, 15, 415, 435], radius=10, fill=(15, 23, 42), outline=(51, 65, 85), width=1)
        
        # Terminal top bar
        draw.rounded_rectangle([15, 15, 415, 42], radius=10, fill=(30, 41, 59))
        draw.rectangle([15, 30, 415, 42], fill=(30, 41, 59))
        draw.ellipse([25, 23, 33, 31], fill=(239, 68, 68))
        draw.ellipse([38, 23, 46, 31], fill=(245, 158, 11))
        draw.ellipse([51, 23, 59, 31], fill=(34, 197, 94))
        draw.text((70, 21), "pistgat-bench runner  --  bash", fill=(148, 163, 184), font=font_mono_small)

        # Command prompt
        draw.text((25, 52), "$ pistgat-bench run --dataset nasa --sample", fill=(56, 189, 248), font=font_mono)

        # Ingestion message (appears after frame 3)
        if f_idx >= 3:
            draw.text((25, 75), "[OK] Loaded NASA Prognostics (5 cells)", fill=(52, 211, 153), font=font_mono_small)
            draw.text((25, 92), "     Cells: B0005, C1, C2, C3, C4", fill=(148, 163, 184), font=font_mono_small)

        # Training phase
        if f_idx >= 7:
            epoch = min(30, int(1 + ((f_idx - 7) / 18) * 30))
            loss_val = max(0.012, 0.285 * math.exp(-epoch / 6.0) + 0.008)
            draw.text((25, 118), f"[1/3] Training PISTGAT: Epoch {epoch}/30", fill=(244, 114, 182), font=font_mono_small)
            # Progress bar
            bar_w = 160
            fill_w = int((epoch / 30) * bar_w)
            draw.rectangle([25, 136, 25 + bar_w, 144], fill=(30, 41, 59), outline=(71, 85, 105))
            draw.rectangle([25, 136, 25 + fill_w, 144], fill=(236, 72, 153))
            draw.text((195, 133), f"Loss: {loss_val:.4f}", fill=(203, 213, 225), font=font_mono_small)

        # Benchmark results table (revealed after frame 22)
        if f_idx >= 22:
            draw.text((25, 160), "Benchmark Results: NASA B0005 (MLflow)", fill=(56, 189, 248), font=font_mono_bold)
            draw.line([25, 178, 400, 178], fill=(71, 85, 105), width=1)
            draw.text((25, 184), "Model     RMSE(%)   MAE(%)   Latency", fill=(148, 163, 184), font=font_mono_small)
            draw.line([25, 198, 400, 198], fill=(51, 65, 85), width=1)

            draw.text((25, 204), "EKF       13.80%    13.79%   0.01 ms", fill=(245, 158, 11), font=font_mono_small)
            draw.text((25, 222), "LSTM       3.22%     3.01%   0.10 ms", fill=(56, 189, 248), font=font_mono_small)
            draw.text((25, 240), "PISTGAT    1.89%     1.44%   0.07 ms", fill=(52, 211, 153), font=font_mono_bold)

            draw.line([25, 258, 400, 258], fill=(51, 65, 85), width=1)
            draw.text((25, 268), "[OK] Telemetry -> web/public/telemetry.json", fill=(52, 211, 153), font=font_mono_small)
            draw.text((25, 286), "[OK] Run logged to MLflow (ID: run_9f82)", fill=(168, 85, 247), font=font_mono_small)

        # -------------------------------------------------------------
        # Right Pane: 3D Pack Imbalance Visualizer (x: 430 to 845, y: 15 to 435)
        # -------------------------------------------------------------
        draw.rounded_rectangle([430, 15, 845, 435], radius=10, fill=(7, 11, 19), outline=(51, 65, 85), width=1)

        # Header of visualizer
        draw.text((445, 26), "PISTGAT 3D Pack Visualizer", fill=(248, 250, 252), font=font_title)
        draw.text((445, 46), f"Active Configuration: 4S1P Pack | Cycle {cycle_current}/60", fill=(148, 163, 184), font=font_sub)
        
        # HUD badges
        pack_soh = max(0.68, 0.98 - (cycle_current / 60) * 0.26)
        imbalance = 0.012 + (cycle_current / 60) * 0.054
        draw.rounded_rectangle([445, 68, 545, 105], radius=6, fill=(15, 23, 42), outline=(34, 197, 94))
        draw.text((452, 72), "PACK SOH", fill=(148, 163, 184), font=font_mono_small)
        draw.text((452, 85), f"{(pack_soh * 100):.1f}%", fill=(34, 197, 94), font=font_mono_bold)

        draw.rounded_rectangle([555, 68, 665, 105], radius=6, fill=(15, 23, 42), outline=(245, 158, 11))
        draw.text((562, 72), "IMBALANCE Δ", fill=(148, 163, 184), font=font_mono_small)
        draw.text((562, 85), f"+{(imbalance * 100):.2f}%", fill=(245, 158, 11), font=font_mono_bold)

        draw.rounded_rectangle([675, 68, 775, 105], radius=6, fill=(15, 23, 42), outline=(56, 189, 248))
        draw.text((682, 72), "MAX TEMP", fill=(148, 163, 184), font=font_mono_small)
        draw.text((682, 85), f"{(25.0 + cycle_current * 0.22):.1f}°C", fill=(56, 189, 248), font=font_mono_bold)

        # 3D Pack Cylinders Drawing with Orbiting angle
        orbit_angle = (f_idx / num_frames) * math.pi * 0.4 - 0.2
        cos_a = math.cos(orbit_angle)
        sin_a = math.sin(orbit_angle)

        center_x, center_y = 635, 220
        cell_spacing = 68
        num_cells = 4

        # Transparent acrylic pack casing
        draw.rounded_rectangle([455, 140, 815, 285], radius=8, fill=None, outline=(51, 65, 85), width=2)
        draw.text((465, 145), "ACRYLIC HOLDER 4S1P", fill=(71, 85, 105), font=font_mono_small)

        # Draw 4 cells
        for c_i in range(num_cells):
            # Calculate position with perspective
            x_rel = (c_i - (num_cells - 1) / 2) * cell_spacing
            cx = center_x + int(x_rel * cos_a)
            cy = center_y + int(x_rel * sin_a * 0.3)

            # Health degradation of this cell
            rate = 1.0 + c_i * 0.18
            c_soh = max(0.60, 0.98 - (cycle_current / 60) * 0.25 * rate)

            # Color interpolation
            if c_soh >= 0.90:
                c_color = (34, 197, 94) # Green
            elif c_soh >= 0.80:
                c_color = (234, 179, 8) # Yellow
            elif c_soh >= 0.72:
                c_color = (249, 115, 22) # Orange
            else:
                c_color = (239, 68, 68) # Red

            # Cylindrical cell body
            w_half, h_half = 22, 48
            draw.rounded_rectangle([cx - w_half, cy - h_half, cx + w_half, cy + h_half], radius=10, fill=c_color, outline=(255, 255, 255), width=1)
            # Metallic top terminal
            draw.rectangle([cx - 8, cy - h_half - 6, cx + 8, cy - h_half], fill=(203, 213, 225), outline=(100, 116, 139))
            
            # Label
            draw.text((cx - 16, cy - 8), f"C{c_i + 1}", fill=(15, 23, 42), font=font_mono_bold)
            draw.text((cx - 20, cy + h_half + 8), f"{(c_soh * 100):.0f}%", fill=c_color, font=font_mono_small)

            # Copper busbar connection between cells
            if c_i < num_cells - 1:
                next_x_rel = (c_i + 1 - (num_cells - 1) / 2) * cell_spacing
                next_cx = center_x + int(next_x_rel * cos_a)
                next_cy = center_y + int(next_x_rel * sin_a * 0.3)
                draw.line([cx, cy - h_half - 3, next_cx, next_cy - h_half - 3], fill=(148, 163, 184), width=4)

        # 2D Heatmap / Degradation Mini Curve at bottom
        draw.line([455, 335, 815, 335], fill=(30, 41, 59), width=1)
        draw.text((455, 345), "Real-Time SOH Fade Trajectory:", fill=(148, 163, 184), font=font_mono_small)

        # Mini curve plot
        curve_pts = []
        for step in range(1, 61):
            px = 455 + int((step / 60) * 360)
            s_val = max(0.60, 0.98 - (step / 60) * 0.28)
            py = 415 - int((s_val - 0.60) / 0.40 * 55)
            curve_pts.append((px, py))
        draw.line(curve_pts, fill=(52, 211, 153), width=2)

        # Current timeline indicator cursor
        cur_px = 455 + int((cycle_current / 60) * 360)
        draw.line([cur_px, 355, cur_px, 420], fill=(56, 189, 248), width=2)

        frames.append(img)

    # Save animated GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_per_frame_ms,
        loop=0,
        optimize=True
    )
    print(f"Generated {output_path} with {len(frames)} frames.")


if __name__ == "__main__":
    create_demo_gif()
