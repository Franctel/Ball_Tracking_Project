"""
Utility functions for generating team wagon radar charts and session-based analysis.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64


# Constants for sample data generation
SAMPLE_DATA_SEED_OUR_TEAM = 42
SAMPLE_DATA_SEED_OPPONENT_TEAM = 84
SAMPLE_DATA_BALL_COUNT = 100


def map_bowling_type_radar(skill):
    """Map bowling skill to simplified bowling type for radar filtering."""
    skill_str = str(skill).lower()
    if "spin" in skill_str or "off" in skill_str or "leg" in skill_str:
        return "Spin"
    elif "pace" in skill_str or "fast" in skill_str or "medium" in skill_str:
        return "Pace"
    else:
        return "Other"


def generate_session_radar_chart(
    ball_by_ball_df,
    day,
    inning,
    session,
    team_name="Team",
    bowler_type=None,
    run_filter=None
):
    """
    Radar-style session wagon wheel chart with optional run filtering.
    Generates 8x8 inch chart at 260 dpi (approximately 2080x2080 pixels).
    """

    df = ball_by_ball_df.copy()

    # ------------------------------
    # NORMALISING run_filter INPUT
    # ------------------------------
    if run_filter is None or str(run_filter).lower() == "all":
        run_set = None
    else:
        try:
            if isinstance(run_filter, (list, tuple, set)):
                run_set = set(int(x) for x in run_filter)
            else:
                s = str(run_filter)
                if "," in s:
                    run_set = set(int(x.strip()) for x in s.split(",") if x.strip())
                else:
                    run_set = {int(s.strip())}
        except (ValueError, TypeError):
            # If conversion fails, treat as no filter
            run_set = None

    # ------------------------------
    # Extract Day / Session columns
    # ------------------------------
    if "Day" not in df.columns or "SessionNo" not in df.columns:
        wide_col = "scrM_IsWideBall" if "scrM_IsWideBall" in df.columns else None
        noball_col = "scrM_IsNoBall" if "scrM_IsNoBall" in df.columns else None
        is_wide_arr = np.array(df[wide_col].fillna(0).astype(int)) if wide_col else np.zeros(len(df))
        is_noball_arr = np.array(df[noball_col].fillna(0).astype(int)) if noball_col else np.zeros(len(df))
        df["__is_legal"] = 1 - (is_wide_arr | is_noball_arr)

        sort_cols = [c for c in ["scrM_InningNo", "scrM_OverNo", "scrM_DelNo"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

        df["__legal_cum"] = df["__is_legal"].cumsum()
        legal_idx_0 = np.maximum(df["__legal_cum"] - 1, 0)
        session_index = (legal_idx_0 // (30 * 6)).astype(int)

        df["Day"] = (session_index // 3) + 1
        df["SessionNo"] = (session_index % 3) + 1

    day_col = "Day"
    session_col = "SessionNo"

    df = df[
        (df[day_col] == day) &
        (df["scrM_InningNo"] == inning) &
        (df[session_col] == session)
    ]

    # ------------------------------
    # RUN FILTER
    # ------------------------------
    if run_set:
        df = df[df["scrM_BatsmanRuns"].isin(run_set)]

    # ------------------------------
    # Bowler Type filter
    # ------------------------------
    if bowler_type:
        if "scrM_BowlerSkill" in df.columns:
            df["BowlingType"] = df["scrM_BowlerSkill"].apply(map_bowling_type_radar)
            df = df[df["BowlingType"] == bowler_type]

    # ------------------------------
    # No Data Chart
    # ------------------------------
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))  # bigger
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines['polar'].set_visible(False)
        ax.text(0.5, 0.5, "No Data", ha="center", va="center",
                transform=ax.transAxes, color="red", fontsize=24, fontweight="bold")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=260, transparent=True)
        plt.close(fig)
        buf.seek(0)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"

    # ------------------------------
    # Sector breakdown
    # ------------------------------
    sectors = ["Mid Wicket", "Square Leg", "Fine Leg", "Third Man",
               "Point", "Covers", "Long Off", "Long On"]
    breakdown_data = [{"1s":0,"2s":0,"3s":0,"4s":0,"6s":0} for _ in sectors]
    sector_map = {name: i for i, name in enumerate(sectors)}

    for _, row in df.iterrows():
        sec = str(row.get("scrM_WagonArea_zName", ""))
        runs = int(row.get("scrM_BatsmanRuns", 0))
        if sec in sector_map and runs > 0:
            idx = sector_map[sec]
            if runs == 1: breakdown_data[idx]["1s"] += 1
            elif runs == 2: breakdown_data[idx]["2s"] += 1
            elif runs == 3: breakdown_data[idx]["3s"] += 1
            elif runs == 4: breakdown_data[idx]["4s"] += 1
            elif runs == 6: breakdown_data[idx]["6s"] += 1

    # ------------------------------
    # Start Plot  (BIGGER SIZE)
    # ------------------------------
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))  # bigger

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines['polar'].set_visible(False)

    scale = 0.9
    ax.set_aspect('equal')

    # ------------------------------
    # Drawing (BIGGER RIM + ELEMENTS)
    # ------------------------------
    rim_radius = 1.10 * scale
    rim_circle = plt.Circle((0, 0), rim_radius, transform=ax.transData._b,
                            color='black', linewidth=26, fill=False,   # thicker rim
                            zorder=5, clip_on=False)
    ax.add_artist(rim_circle)

    ax.add_artist(plt.Circle((0, 0), 1.0 * scale, transform=ax.transData._b,
                             color='#19a94b', zorder=0))
    ax.add_artist(plt.Circle((0, 0), 0.6 * scale, transform=ax.transData._b,
                             color='#4CAF50', zorder=1))
    ax.add_artist(plt.Rectangle((-0.08 * scale / 2, -0.33 * scale / 2),
                                0.08 * scale, 0.33 * scale,
                                transform=ax.transData._b, color='burlywood', zorder=2))

    for angle in np.linspace(0, 2*np.pi, 9):
        ax.plot([angle, angle], [0, 1.0 * scale],
                color='white', linewidth=3, zorder=3)

    # ------------------------------
    # Sector totals + highlight
    # ------------------------------
    sector_runs = [(bd["1s"] + bd["2s"]*2 + bd["3s"]*3 +
                    bd["4s"]*4 + bd["6s"]*6) for bd in breakdown_data]

    total_runs = sum(sector_runs)

    if total_runs > 0:
        max_idx = sector_runs.index(max(sector_runs))
        sector_angles_deg = [112.5, 67.5, 22.5, 337.5,
                             292.5, 247.5, 202.5, 157.5]
        ax.bar(np.deg2rad(sector_angles_deg[max_idx]), 1.0 * scale,
               width=np.radians(45), color='red', alpha=0.25, zorder=1)

    # ------------------------------
    # Fielding labels (BIGGER)
    # ------------------------------
    position_labels = [
        ("Mid Wicket", 112.5, -110, -0.02),
        ("Square Leg", 67.5, -70, -0.02),
        ("Fine Leg", 22.5, -25, 0.00),
        ("Third Man", 337.5, 20, -0.02),
        ("Point", 292.5, 70, -0.01),
        ("Covers", 247.5, 110, -0.01),
        ("Long Off", 202.5, 155, -0.02),
        ("Long On", 157.5, 200, -0.02)
    ]

    for text, angle_deg, rotation_deg, dist_offset in position_labels:
        rad = np.deg2rad(angle_deg)
        ax.text(rad, rim_radius + dist_offset, text,
                color='white', fontsize=16, fontweight='bold',   # bigger labels
                ha='center', va='center', rotation=rotation_deg,
                rotation_mode='anchor', zorder=6)

    # ------------------------------
    # Runs + Percentage text (BIGGER)
    # ------------------------------
    box_positions = [
        (103.5, 0, 0.70), (67.5, 0, 0.70),
        (22.5, 0, 0.80), (337.5, 0, 0.80),
        (295.5, 0, 0.75), (250.5, 0, 0.70),
        (204.5, 1, 0.59), (155.5, 1, 0.59)
    ]

    for i, (angle_deg, rot, dist) in enumerate(box_positions):
        rad = np.deg2rad(angle_deg)
        r = dist * scale
        runs = sector_runs[i]
        pct = (runs / total_runs * 100) if total_runs > 0 else 0

        ax.text(rad, r,
                f"{runs}\n({pct:.1f}%)",
                color='white', fontsize=19, fontweight='bold',   # bigger text
                ha='center', va='center',
                rotation=0,
                linespacing=1.15)

    # ------------------------------
    # EXPORT (BIG)
    # ------------------------------
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=260, transparent=True)
    plt.close(fig)
    buf.seek(0)

    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


def generate_team_wagon_radar(team_name, df, mode="batting", stance=None, size_inches=8, dpi=260):
    """
    Generate the wagon/radar image for a team aggregated across rows in df.
    - team_name: string (team to compute for)
    - df: ball-by-ball pandas DataFrame (should contain scrM_WagonArea_zName, scrM_BatsmanRuns,
          scrM_IsBoundry, scrM_IsSixer, scrM_tmMIdBattingName, scrM_tmMIdBowlingName)
    - mode: "batting" or "bowling"
      * batting => count runs scored by selected team (scrM_tmMIdBattingName == team_name)
      * bowling => count runs conceded by selected team (scrM_tmMIdBowlingName == team_name)
    - stance: None or "LHB" to mirror labels (keeps compatibility)
    Returns: "data:image/png;base64,..." string (PNG)
    """

    # expected areas (the radar geometry uses this order for RHB orientation)
    labels_expected = ["Mid Wicket","Square Leg","Fine Leg","Third Man","Point","Covers","Long Off","Long On"]

    # If df empty or None, return blank image with text
    if df is None or df.empty:
        # create blank placeholder image
        fig, ax = plt.subplots(figsize=(size_inches, size_inches))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=18)
        ax.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=dpi, transparent=True)
        plt.close(fig)
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")

    # select rows depending on mode
    # df already filtered by caller (team OR opponents)
    sel = df.copy()

    # normalize area column name
    area_col = "scrM_WagonArea_zName"
    runs_col = "scrM_BatsmanRuns"

    # If column not found, return placeholder
    if area_col not in sel.columns or runs_col not in sel.columns:
        # produce placeholder
        fig, ax = plt.subplots(figsize=(size_inches, size_inches))
        ax.text(0.5, 0.5, "Wagon area / runs columns missing", ha="center", va="center", fontsize=12)
        ax.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=dpi, transparent=True)
        plt.close(fig)
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")

    # Clean strings
    sel[area_col] = sel[area_col].astype(str).str.strip()

    # Map canonical DB area names to expected labels.
    # The DB contains: Covers, Fine Leg, Long Off, Long On, Mid Wicket, Point, Square Leg, Third Man
    # We need to reorder into labels_expected.
    # Build a small aggregation keyed by canonical names (DB strings)
    agg = {}
    for lab in labels_expected:
        agg[lab] = {"runs": 0, "1s":0, "2s":0, "3s":0, "4s":0, "6s":0, "balls":0}

    # Build mapping from DB values to our expected labels (handle minor variants)
    db_to_expected = {
        "Covers": "Covers",
        "Fine Leg": "Fine Leg",
        "Long Off": "Long Off",
        "Long On": "Long On",
        "Mid Wicket": "Mid Wicket",
        "Point": "Point",
        "Square Leg": "Square Leg",
        "Third Man": "Third Man",
        # tolerant matches:
        "Cover": "Covers",
        "Fine-Leg": "Fine Leg",
        "Long-On": "Long On",
        "Long-Off": "Long Off",
        "MidWicket": "Mid Wicket",
        "ThirdMan": "Third Man"
    }

    # iterate rows and add to respective bin
    for _, row in sel.iterrows():
        a = str(row.get(area_col, "")).strip()
        if not a:
            continue
        mapped = db_to_expected.get(a, None)
        # try case-insensitive match fallback
        if mapped is None:
            for k,v in db_to_expected.items():
                if k.lower() == a.lower():
                    mapped = v
                    break
        if mapped is None:
            # area not recognized — skip
            continue

        runs_val = 0
        try:
            runs_val = int(row.get(runs_col, 0) or 0)
        except Exception:
            try:
                runs_val = int(float(row.get(runs_col, 0) or 0))
            except Exception:
                runs_val = 0

        agg[mapped]["runs"] += runs_val
        agg[mapped]["balls"] += 1

        # count 1s/2s/3s by runs equality
        if runs_val == 1:
            agg[mapped]["1s"] += 1
        elif runs_val == 2:
            agg[mapped]["2s"] += 1
        elif runs_val == 3:
            agg[mapped]["3s"] += 1

        # boundaries detection if flags present
        if "scrM_IsBoundry" in sel.columns:
            try:
                if int(row.get("scrM_IsBoundry") or 0) == 1:
                    agg[mapped]["4s"] += 1
            except Exception:
                pass
        else:
            # fallback: if runs_val == 4 and not flagged
            if runs_val == 4:
                agg[mapped]["4s"] += 1

        if "scrM_IsSixer" in sel.columns:
            try:
                if int(row.get("scrM_IsSixer") or 0) == 1:
                    agg[mapped]["6s"] += 1
            except Exception:
                pass
        else:
            if runs_val == 6:
                agg[mapped]["6s"] += 1

    # prepare lists in labels_expected order
    sector_runs = [agg[l]["runs"] for l in labels_expected]
    breakdown_data = [{"1s":agg[l]["1s"], "2s":agg[l]["2s"], "3s":agg[l]["3s"], "4s":agg[l]["4s"], "6s":agg[l]["6s"]} for l in labels_expected]
    total_sector_runs = sum(sector_runs)

    # --------------- Now plotting (wagon style) ----------------
    num_vars = 8
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(size_inches, size_inches), subplot_kw=dict(polar=True))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    # hide spine
    try:
        ax.spines['polar'].set_visible(False)
    except Exception:
        pass

    # Scaling
    scale = 0.9
    ax.set_aspect('equal')

    # Black rim (radius relative)
    rim_radius = 1.10 * scale
    rim_circle = plt.Circle(
        (0, 0),
        rim_radius,
        transform=ax.transData._b,
        color='black',
        linewidth=26,
        fill=False,
        zorder=5,
        clip_on=False
    )
    ax.add_artist(rim_circle)

    # Ground circles (green)
    outer_circle = plt.Circle((0, 0), 1.0 * scale, transform=ax.transData._b, color='#19a94b', zorder=0)
    inner_circle = plt.Circle((0, 0), 0.6 * scale, transform=ax.transData._b, color='#4CAF50', zorder=1)
    ax.add_artist(outer_circle)
    ax.add_artist(inner_circle)

    # Pitch rectangle
    pitch_width = 0.08 * scale
    pitch_height = 0.33 * scale
    pitch_x = -pitch_width / 2
    pitch_y = -pitch_height / 2
    pitch = plt.Rectangle((pitch_x, pitch_y), pitch_width, pitch_height, transform=ax.transData._b, color='burlywood', zorder=2)
    ax.add_artist(pitch)

    # Sector lines (every 45°)
    for angle in np.linspace(0, 2 * np.pi, 9):
        ax.plot([angle, angle], [0, 1.0 * scale], color='white', linewidth=3, zorder=3)

    # RHB sector angles used in your previous design
    sector_angles_deg = [112.5, 67.5, 22.5, 337.5, 292.5, 247.5, 202.5, 157.5]

    # Highlight max sector
    if total_sector_runs > 0:
        max_idx = int(np.argmax(sector_runs))
        max_angle = np.deg2rad(sector_angles_deg[max_idx])
        ax.bar(max_angle, 1.0 * scale, width=np.radians(45), color='red', alpha=0.25, zorder=1)

    # Position labels (RHB)
    position_labels = [
        ("Mid Wicket", 112.5, -110, -0.02),
        ("Square Leg", 67.5, -70, -0.02),
        ("Fine Leg", 22.5, -25, 0.00),
        ("Third Man", 337.5, 20, -0.02),
        ("Point", 292.5, 70, -0.01),
        ("Covers", 247.5, 110, -0.01),
        ("Long Off", 202.5, 155, -0.02),
        ("Long On", 157.5, 200, -0.02)
    ]

    for text, angle_deg, rotation_deg, dist_offset in position_labels:
        rad = np.deg2rad(angle_deg)
        ax.text(
            rad,
            rim_radius + dist_offset,
            text,
            color='white',
            fontsize=16,
            fontweight='bold',
            ha='center',
            va='center',
            rotation=rotation_deg,
            rotation_mode='anchor',
            zorder=6
        )

    # Boxes with runs and % values (positions tuned to match style)
    box_positions = [
        (103.5, 0, 0.70),
        (67.5, 0, 0.70),
        (22.5, 0, 0.80),
        (337.5, 0, 0.80),
        (295.5, 0, 0.75),
        (250.5, 0, 0.70),
        (204.5, 1, 0.59),
        (155.5, 1, 0.59)
    ]

    for i, (angle_deg, rotation_deg, dist_offset) in enumerate(box_positions):
        rad = np.deg2rad(angle_deg)
        r = dist_offset * scale
        runs = sector_runs[i]
        percentage = (runs / total_sector_runs * 100) if total_sector_runs > 0 else 0
        label_text = f"{runs}\n({percentage:.1f}%)"
        ax.text(
            rad, r,
            label_text,
            color='white',
            fontsize=19,
            fontweight='bold',
            ha='center',
            va='center',
            rotation=0,
            linespacing=1.15
        )

    # Breakdown small text under boxes
    detail_positions = [
        (116.5, 0, 0.68),
        (78.5, 0, 0.68),
        (29.5, 0, 0.70),
        (330.5, 0, 0.70),
        (283.5, 0, 0.68),
        (239.5, 0, 0.72),
        (200.5, 1, 0.72),
        (163.5, 1, 0.70)
    ]

    for i, (angle_deg, rotation_deg, dist_offset) in enumerate(detail_positions):
        rad = np.deg2rad(angle_deg)
        r = dist_offset * scale
        bd = breakdown_data[i]
        breakdown_text = f"1s:{bd['1s']}  2s:{bd['2s']}\n4s:{bd['4s']}  6s:{bd['6s']}"
        ax.text(
            rad, r,
            breakdown_text,
            color='white',
            fontsize=10,
            ha='center',
            va='center',
            rotation=rotation_deg,
            rotation_mode='anchor',
            zorder=11
        )

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', dpi=dpi, transparent=True)
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def create_sample_ball_by_ball_data(team_type, run_filters, bowler_filters):
    """
    Create sample ball-by-ball data for demonstration.
    In a real application, this would query from a database with filters applied.
    """
    # Sample data structure
    areas = ["Mid Wicket", "Square Leg", "Fine Leg", "Third Man", "Point", "Covers", "Long Off", "Long On"]
    
    # Generate random sample data with filters applied
    data_rows = []
    seed = SAMPLE_DATA_SEED_OUR_TEAM if team_type == 'our' else SAMPLE_DATA_SEED_OPPONENT_TEAM
    np.random.seed(seed)
    
    # Only generate data if we have filters
    if not run_filters:
        return pd.DataFrame()
    
    for _ in range(SAMPLE_DATA_BALL_COUNT):
        area = np.random.choice(areas)
        runs = int(np.random.choice(run_filters))
        
        row = {
            'scrM_WagonArea_zName': area,
            'scrM_BatsmanRuns': runs,
            'scrM_IsBoundry': 1 if runs == 4 else 0,
            'scrM_IsSixer': 1 if runs == 6 else 0,
            'scrM_tmMIdBattingName': f"{team_type.capitalize()} Team",
            'scrM_tmMIdBowlingName': "Opponent Team" if team_type == 'our' else "Our Team",
            'scrM_BowlerSkill': np.random.choice(['Pace', 'Spin', 'Other'])
        }
        
        # Apply bowler filter
        if row['scrM_BowlerSkill'] in bowler_filters:
            data_rows.append(row)
    
    if not data_rows:
        return pd.DataFrame()
    
    return pd.DataFrame(data_rows)
