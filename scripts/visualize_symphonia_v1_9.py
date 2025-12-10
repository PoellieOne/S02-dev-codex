#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_symphonia_v1_9.py

Uitgebreide visualisatie toolkit voor Symphonia S02 bench data.
Geoptimaliseerd voor v1.9 Canonical Architecture output.

Nieuwe features t.o.v. v1.4:
- Hiërarchische RotorState visualisatie (STILL → MOVEMENT)
- Unlock mechanisme tracking
- Pure stereo tile marking
- CompassSign v3 confidence

Gebruik:
    python3 visualize_symphonia_v1_9.py <csv_file> [--output-dir <dir>] [--show]

Voorbeeld:
    python3 visualize_symphonia_v1_9.py core0_events_v1_9_bench_tolerant.csv --show
    python3 visualize_symphonia_v1_9.py core0_events_v1_9_bench_tolerant.csv --output-dir ./plots

Gegenereerde plots:
    1. overview_dashboard.png      - 4-panel overzicht
    2. state_machine.png           - Hiërarchische state visualisatie (NIEUW)
    3. cycles_and_rotations.png    - Cycle accumulatie en rotaties
    4. compass_and_lock.png        - Kompas progressie en lock transitie
    5. rpm_analysis.png            - RPM en jitter analyse
    6. phase_distribution.png      - Sensor A/B cycle distributie
    7. awareness_timeline.png      - Awareness confidence tijdlijn
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Configuratie
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'  
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['font.size'] = 10


def load_data(csv_path: str) -> pd.DataFrame:
    """Laad en prepareer de CSV data."""
    df = pd.read_csv(csv_path)
    
    # Bereken afgeleide kolommen indien nodig
    if 't_us' in df.columns:
        df['t_s'] = (df['t_us'] - df['t_us'].iloc[0]) / 1e6  # Relatieve tijd in seconden
    
    if 'tile_index' in df.columns:
        df['tile_seq'] = range(len(df))  # Sequentiële index voor plotting
    
    return df


def plot_overview_dashboard(df: pd.DataFrame, output_path: Path = None, show: bool = False):
    """
    Plot 1: 4-panel overzicht dashboard
    - Rotaties over tijd
    - RPM over tijd
    - Direction lock state
    - Awareness confidence
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # Kleuren per state
    lock_colors = {
        'UNLOCKED': '#ffcccc',
        'SOFT_LOCK': '#ffffcc', 
        'LOCKED': '#ccffcc'
    }
    rotor_colors = {
        'STILL': '#e0e0e0',
        'MOVEMENT': '#ffffff'
    }
    
    # Panel 1: Rotaties over tijd
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df['t_s'], df['rotations'], 'b-', linewidth=2, label='Rotaties (signed)')
    if 'total_cycles_physical' in df.columns:
        ax1.plot(df['t_s'], df['total_cycles_physical'] / 12, 'g--', 
                 linewidth=1.5, alpha=0.7, label='Cycles/12 (fysiek)')
    
    # Markeer lock transitie
    if 'direction_lock_state' in df.columns:
        first_lock = df[df['direction_lock_state'] == 'LOCKED']
        if len(first_lock) > 0:
            lock_t = first_lock.iloc[0]['t_s']
            ax1.axvline(x=lock_t, color='green', linestyle=':', alpha=0.7, label='LOCK moment')
    
    # v1.9: Markeer STILL → MOVEMENT transitie
    if 'rotor_state' in df.columns:
        first_movement = df[df['rotor_state'] == 'MOVEMENT']
        if len(first_movement) > 0:
            move_t = first_movement.iloc[0]['t_s']
            ax1.axvline(x=move_t, color='blue', linestyle=':', alpha=0.7, label='MOVEMENT start')
    
    ax1.set_xlabel('Tijd (s)')
    ax1.set_ylabel('Rotaties')
    ax1.set_title('Rotatie Progressie')
    ax1.legend(loc='upper left', fontsize=8)
    
    # Panel 2: RPM over tijd
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df['t_s'], df['rpm_est'], 'b-', linewidth=2, label='RPM (smoothed)')
    ax2.plot(df['t_s'], df['rpm_inst'], 'c.', markersize=3, alpha=0.5, label='RPM (instant)')
    ax2.axhline(y=60, color='gray', linestyle='--', alpha=0.5, label='60 RPM ref')
    ax2.set_xlabel('Tijd (s)')
    ax2.set_ylabel('RPM')
    ax2.set_title('Snelheid (RPM)')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_ylim(bottom=0)
    
    # Panel 3: Direction en Lock state (met rotor_state achtergrond)
    ax3 = fig.add_subplot(gs[1, 0])
    
    # v1.9: Rotor state achtergrond eerst
    if 'rotor_state' in df.columns:
        prev_state = None
        prev_t = df['t_s'].iloc[0]
        for idx, row in df.iterrows():
            if row['rotor_state'] != prev_state and prev_state is not None:
                color = rotor_colors.get(prev_state, 'white')
                ax3.axvspan(prev_t, row['t_s'], alpha=0.2, color=color)
                prev_t = row['t_s']
            prev_state = row['rotor_state']
        if prev_state:
            color = rotor_colors.get(prev_state, 'white')
            ax3.axvspan(prev_t, df['t_s'].iloc[-1], alpha=0.2, color=color)
    
    # Plot compass scores
    if 'compass_global_score' in df.columns:
        ax3.plot(df['t_s'], df['compass_global_score'], 'b-', linewidth=2, label='Global score')
    if 'compass_window_score' in df.columns:
        ax3.plot(df['t_s'], df['compass_window_score'], 'c--', linewidth=1.5, alpha=0.7, label='Window conf')
    
    # Lock state achtergrond (bovenop rotor state)
    if 'direction_lock_state' in df.columns:
        prev_state = None
        prev_t = df['t_s'].iloc[0]
        for idx, row in df.iterrows():
            if row['direction_lock_state'] != prev_state and prev_state is not None:
                color = lock_colors.get(prev_state, 'white')
                ax3.axvspan(prev_t, row['t_s'], alpha=0.3, color=color)
                prev_t = row['t_s']
            prev_state = row['direction_lock_state']
        if prev_state:
            color = lock_colors.get(prev_state, 'white')
            ax3.axvspan(prev_t, df['t_s'].iloc[-1], alpha=0.3, color=color)
    
    # Thresholds
    ax3.axhline(y=0.45, color='green', linestyle=':', alpha=0.5, label='threshold_high')
    ax3.axhline(y=-0.45, color='green', linestyle=':', alpha=0.5)
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    ax3.set_xlabel('Tijd (s)')
    ax3.set_ylabel('Score')
    ax3.set_title('Kompas & Direction Lock (v1.9)')
    ax3.legend(loc='upper left', fontsize=8)
    ax3.set_ylim(-1.1, 1.1)
    
    # Panel 4: Awareness confidence
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.fill_between(df['t_s'], 0, df['awareness_conf'], alpha=0.3, color='purple')
    ax4.plot(df['t_s'], df['awareness_conf'], 'purple', linewidth=2, label='Awareness')
    if 'motion_conf' in df.columns:
        ax4.plot(df['t_s'], df['motion_conf'], 'orange', linewidth=1.5, 
                 alpha=0.7, linestyle='--', label='Motion conf')
    ax4.set_xlabel('Tijd (s)')
    ax4.set_ylabel('Confidence')
    ax4.set_title('Awareness & Motion Confidence')
    ax4.legend(loc='upper left', fontsize=8)
    ax4.set_ylim(0, 1.1)
    
    # Hoofdtitel
    direction = df['direction_global_effective'].iloc[-1] if 'direction_global_effective' in df.columns else '?'
    rotations = df['rotations'].iloc[-1] if 'rotations' in df.columns else 0
    lock_state = df['direction_lock_state'].iloc[-1] if 'direction_lock_state' in df.columns else '?'
    fig.suptitle(f'Symphonia S02 v1.9 — {direction}, {rotations:.2f} rotaties, {lock_state}', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path / 'overview_dashboard.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ overview_dashboard.png")
    if show:
        plt.show()
    plt.close()


def plot_state_machine(df: pd.DataFrame, output_path: Path = None, show: bool = False):
    """
    Plot 2 (NIEUW): Hiërarchische state machine visualisatie
    - RotorState (STILL/MOVEMENT)
    - DirectionLockState (UNLOCKED/SOFT_LOCK/LOCKED)
    - State transitions annotaties
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    
    x = df['tile_seq']
    
    # Panel 1: RotorState
    ax1 = axes[0]
    rotor_map = {'STILL': 0, 'MOVEMENT': 1}
    if 'rotor_state' in df.columns:
        rotor_num = df['rotor_state'].map(rotor_map).fillna(0)
        ax1.fill_between(x, 0, rotor_num, alpha=0.5, 
                        color='blue', step='post')
        ax1.step(x, rotor_num, where='post', color='darkblue', linewidth=2)
        
        # Annoteer transities
        prev = None
        for idx, row in df.iterrows():
            curr = row['rotor_state']
            if curr != prev and prev is not None:
                ax1.annotate(f'→{curr}', 
                           xy=(row['tile_seq'], rotor_map.get(curr, 0)),
                           fontsize=8, color='darkblue')
            prev = curr
    
    ax1.set_ylabel('RotorState')
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['STILL', 'MOVEMENT'])
    ax1.set_ylim(-0.2, 1.4)
    ax1.set_title('Hiërarchische State Machine (v1.9)')
    
    # Panel 2: DirectionLockState
    ax2 = axes[1]
    lock_map = {'UNLOCKED': 0, 'SOFT_LOCK': 1, 'LOCKED': 2}
    lock_colors_fill = {'UNLOCKED': 'red', 'SOFT_LOCK': 'orange', 'LOCKED': 'green'}
    
    if 'direction_lock_state' in df.columns:
        lock_num = df['direction_lock_state'].map(lock_map).fillna(0)
        
        # Gekleurde achtergrond per state
        prev_state = None
        prev_x = x.iloc[0]
        for i, (idx, row) in enumerate(df.iterrows()):
            curr = row['direction_lock_state']
            if curr != prev_state and prev_state is not None:
                color = lock_colors_fill.get(prev_state, 'gray')
                ax2.axvspan(prev_x, row['tile_seq'], alpha=0.3, color=color)
                prev_x = row['tile_seq']
            prev_state = curr
        if prev_state:
            color = lock_colors_fill.get(prev_state, 'gray')
            ax2.axvspan(prev_x, x.iloc[-1], alpha=0.3, color=color)
        
        ax2.step(x, lock_num, where='post', color='black', linewidth=2)
        
        # Annoteer transities
        prev = None
        for idx, row in df.iterrows():
            curr = row['direction_lock_state']
            if curr != prev and prev is not None:
                ax2.annotate(f'→{curr}', 
                           xy=(row['tile_seq'], lock_map.get(curr, 0)),
                           fontsize=7, color='black', fontweight='bold')
            prev = curr
    
    ax2.set_ylabel('LockState')
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['UNLOCKED', 'SOFT_LOCK', 'LOCKED'])
    ax2.set_ylim(-0.2, 2.6)
    
    # Panel 3: Compass scores
    ax3 = axes[2]
    ax3.plot(x, df['compass_global_score'], 'b-', linewidth=2, label='Global score')
    ax3.plot(x, df['compass_window_score'], 'c--', linewidth=1.5, alpha=0.7, label='Window conf')
    ax3.axhline(y=0.45, color='green', linestyle=':', alpha=0.7, label='threshold_high')
    ax3.axhline(y=-0.45, color='green', linestyle=':', alpha=0.7)
    ax3.axhline(y=0.15, color='orange', linestyle=':', alpha=0.5, label='threshold_low')
    ax3.axhline(y=-0.15, color='orange', linestyle=':', alpha=0.5)
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax3.set_ylabel('Score')
    ax3.legend(loc='upper left', fontsize=7)
    ax3.set_ylim(-1.1, 1.1)
    
    # Panel 4: Pure stereo marking
    ax4 = axes[3]
    if 'is_pure_stereo' in df.columns:
        stereo = df['is_pure_stereo'].astype(int)
        ax4.bar(x, stereo, color='purple', alpha=0.7, width=1.0)
        ax4.set_ylabel('Pure Stereo')
        ax4.set_ylim(0, 1.5)
        
        # Toon ratio
        ratio = stereo.sum() / len(stereo) * 100
        ax4.set_title(f'Pure Stereo Tiles ({ratio:.0f}% van totaal)')
    
    ax4.bar(x, df['nA'] / df[['nA', 'nB']].max().max(), color='blue', alpha=0.3, width=1.0, label='nA')
    ax4.bar(x, -df['nB'] / df[['nA', 'nB']].max().max(), color='red', alpha=0.3, width=1.0, label='nB')
    
    ax4.set_xlabel('Tile sequence')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path / 'state_machine.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ state_machine.png")
    if show:
        plt.show()
    plt.close()


def plot_cycles_and_rotations(df: pd.DataFrame, output_path: Path = None, show: bool = False):
    """
    Plot 3: Gedetailleerde cycle en rotatie analyse
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    x = df['tile_seq']
    
    # Panel 1: Cycles per tile
    ax1 = axes[0]
    
    # Kleur op basis van state
    colors = []
    for _, row in df.iterrows():
        if row.get('tile_state') == 'BOOT':
            colors.append('gray')
        elif row.get('rotor_state') == 'STILL':
            colors.append('lightblue')
        elif row.get('direction_lock_state') == 'LOCKED':
            colors.append('green')
        elif row.get('direction_lock_state') == 'SOFT_LOCK':
            colors.append('orange')
        else:
            colors.append('red')
    
    ax1.bar(x, df['cycles_physical'], color=colors, alpha=0.7, width=1.0)
    ax1.set_ylabel('Cycles per tile')
    ax1.set_title('Cycles Physical per Tile')
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    
    # Legend
    patches = [
        mpatches.Patch(color='gray', alpha=0.7, label='BOOT'),
        mpatches.Patch(color='lightblue', alpha=0.7, label='STILL'),
        mpatches.Patch(color='red', alpha=0.7, label='UNLOCKED'),
        mpatches.Patch(color='orange', alpha=0.7, label='SOFT_LOCK'),
        mpatches.Patch(color='green', alpha=0.7, label='LOCKED'),
    ]
    ax1.legend(handles=patches, loc='upper right', fontsize=7)
    
    # Panel 2: Cumulatieve cycles
    ax2 = axes[1]
    cumsum_physical = df['cycles_physical'].cumsum()
    ax2.plot(x, cumsum_physical, 'g-', linewidth=2, label='Σ cycles_physical')
    ax2.plot(x, abs(df['cycle_index']), 'b-', linewidth=2, label='|cycle_index|')
    
    # Markeer claim moment
    if 'cycles_claimed_at_lock' in df.columns:
        claimed = df['cycles_claimed_at_lock'].iloc[-1]
        if claimed > 0:
            locked = df[df['direction_lock_state'] == 'LOCKED']
            if len(locked) > 0:
                lock_x = locked.iloc[0]['tile_seq']
                ax2.axvline(x=lock_x, color='green', linestyle=':', alpha=0.7)
                ax2.annotate(f'Claim: {claimed:.1f}', xy=(lock_x, claimed),
                           fontsize=9, color='green')
    
    ax2.set_ylabel('Cycles')
    ax2.set_title('Cycle Accumulatie & Claim-at-Lock')
    ax2.legend(loc='upper left', fontsize=8)
    
    # Panel 3: Rotaties
    ax3 = axes[2]
    ax3.plot(x, df['rotations'], 'b-', linewidth=2, label='rotations (signed)')
    ax3.plot(x, cumsum_physical / 12, 'g--', linewidth=1.5, alpha=0.7, label='physical/12')
    
    final_rot = df['rotations'].iloc[-1]
    ax3.axhline(y=final_rot, color='blue', linestyle=':', alpha=0.5)
    ax3.annotate(f'{final_rot:.2f}', xy=(x.iloc[-1], final_rot),
                fontsize=10, color='blue', fontweight='bold')
    
    ax3.set_xlabel('Tile sequence')
    ax3.set_ylabel('Rotaties')
    ax3.set_title('Rotatie Progressie')
    ax3.legend(loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path / 'cycles_and_rotations.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ cycles_and_rotations.png")
    if show:
        plt.show()
    plt.close()


def plot_compass_and_lock(df: pd.DataFrame, output_path: Path = None, show: bool = False):
    """
    Plot 4: Kompas en lock analyse
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    x = df['tile_seq']
    
    # Panel 1: Direction codes
    ax1 = axes[0]
    dir_map = {'CW': 1, 'CCW': -1, 'UNDECIDED': 0}
    
    if 'compass_global_direction' in df.columns:
        global_dir = df['compass_global_direction'].map(dir_map).fillna(0)
        ax1.step(x, global_dir, where='post', color='blue', linewidth=2, label='Global dir')
    
    if 'compass_window_direction' in df.columns:
        window_dir = df['compass_window_direction'].map(dir_map).fillna(0)
        ax1.step(x, window_dir, where='post', color='cyan', linewidth=1.5, 
                alpha=0.7, linestyle='--', label='Window dir')
    
    ax1.set_ylabel('Direction')
    ax1.set_yticks([-1, 0, 1])
    ax1.set_yticklabels(['CCW', 'UNDEC', 'CW'])
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_title('Direction Detection')
    
    # Panel 2: Scores met thresholds
    ax2 = axes[1]
    ax2.plot(x, df['compass_global_score'], 'b-', linewidth=2, label='Global score')
    ax2.plot(x, df['compass_window_score'], 'c--', linewidth=1.5, alpha=0.7, label='Window conf')
    
    # Thresholds (bench_tolerant values)
    ax2.axhline(y=0.45, color='green', linestyle=':', alpha=0.7, label='threshold_high (0.45)')
    ax2.axhline(y=-0.45, color='green', linestyle=':', alpha=0.7)
    ax2.axhline(y=0.15, color='orange', linestyle=':', alpha=0.5, label='threshold_low (0.15)')
    ax2.axhline(y=-0.15, color='orange', linestyle=':', alpha=0.5)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    ax2.set_ylabel('Score')
    ax2.set_ylim(-1.1, 1.1)
    ax2.legend(loc='upper left', fontsize=7)
    ax2.set_title('Compass Scores & Thresholds')
    
    # Panel 3: Lock confidence accumulatie
    ax3 = axes[2]
    if 'direction_locked_conf' in df.columns:
        ax3.plot(x, df['direction_locked_conf'], 'g-', linewidth=2, label='Locked conf')
    
    # Lock state achtergrond
    lock_colors = {'UNLOCKED': '#ffcccc', 'SOFT_LOCK': '#ffffcc', 'LOCKED': '#ccffcc'}
    prev_state = None
    prev_x = x.iloc[0]
    for _, row in df.iterrows():
        curr = row['direction_lock_state']
        if curr != prev_state and prev_state is not None:
            color = lock_colors.get(prev_state, 'white')
            ax3.axvspan(prev_x, row['tile_seq'], alpha=0.3, color=color)
            prev_x = row['tile_seq']
        prev_state = curr
    if prev_state:
        color = lock_colors.get(prev_state, 'white')
        ax3.axvspan(prev_x, x.iloc[-1], alpha=0.3, color=color)
    
    ax3.set_xlabel('Tile sequence')
    ax3.set_ylabel('Confidence')
    ax3.set_title('Lock State & Confidence')
    ax3.legend(loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path / 'compass_and_lock.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ compass_and_lock.png")
    if show:
        plt.show()
    plt.close()


def plot_rpm_analysis(df: pd.DataFrame, output_path: Path = None, show: bool = False):
    """
    Plot 5: RPM en cadence analyse
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    x = df['tile_seq']
    
    # Panel 1: RPM met jitter band
    ax1 = axes[0]
    ax1.plot(x, df['rpm_est'], 'b-', linewidth=2, label='RPM (EMA)')
    ax1.fill_between(x, 
                     df['rpm_est'] * (1 - df['rpm_jitter']),
                     df['rpm_est'] * (1 + df['rpm_jitter']),
                     alpha=0.2, color='blue', label='Jitter band')
    ax1.scatter(x, df['rpm_inst'], c='cyan', s=15, alpha=0.5, label='RPM instant')
    
    mean_rpm = df['rpm_est'].mean()
    ax1.axhline(y=mean_rpm, color='gray', linestyle='--', alpha=0.5, 
               label=f'Mean: {mean_rpm:.1f}')
    
    ax1.set_ylabel('RPM')
    ax1.set_title('RPM Analyse')
    ax1.legend(loc='upper right', fontsize=7)
    ax1.set_ylim(bottom=0)
    
    # Panel 2: Jitter
    ax2 = axes[1]
    ax2.fill_between(x, 0, df['rpm_jitter'], alpha=0.5, color='orange')
    ax2.plot(x, df['rpm_jitter'], 'orange', linewidth=2)
    ax2.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='Max jitter (0.6)')
    ax2.set_ylabel('Jitter')
    ax2.set_title('RPM Jitter')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_ylim(0, 1.1)
    
    # Panel 3: Cadence OK
    ax3 = axes[2]
    cadence_ok = df['cadence_ok'].astype(int)
    ax3.fill_between(x, 0, cadence_ok, alpha=0.5, color='green', step='post')
    ax3.step(x, cadence_ok, where='post', color='darkgreen', linewidth=2)
    
    # Motion state overlay
    motion_map = {'STATIC': 0, 'EVALUATING': 0.5, 'MOVING': 1}
    if 'motion_state' in df.columns:
        motion_num = df['motion_state'].map(motion_map).fillna(0)
        ax3.plot(x, motion_num, 'b--', linewidth=1.5, alpha=0.7, label='Motion state')
    
    ax3.set_xlabel('Tile sequence')
    ax3.set_ylabel('OK / Motion')
    ax3.set_title('Cadence OK & Motion State')
    ax3.set_ylim(-0.1, 1.3)
    ax3.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path / 'rpm_analysis.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ rpm_analysis.png")
    if show:
        plt.show()
    plt.close()


def plot_phase_distribution(df: pd.DataFrame, output_path: Path = None, show: bool = False):
    """
    Plot 6: Sensor A/B distributie
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    x = df['tile_seq']
    
    # Panel 1: nA en nB per tile
    ax1 = axes[0]
    width = 0.4
    ax1.bar(x - width/2, df['nA'], width, label='Sensor A', color='blue', alpha=0.7)
    ax1.bar(x + width/2, df['nB'], width, label='Sensor B', color='red', alpha=0.7)
    
    # Markeer pure stereo
    if 'is_pure_stereo' in df.columns:
        stereo_tiles = df[df['is_pure_stereo'] == True]['tile_seq']
        for st in stereo_tiles:
            ax1.axvline(x=st, color='purple', linestyle=':', alpha=0.3)
    
    ax1.set_xlabel('Tile sequence')
    ax1.set_ylabel('Cycles per sensor')
    ax1.set_title('Sensor Cycles per Tile (paarse lijnen = pure stereo)')
    ax1.legend(loc='upper right', fontsize=8)
    
    # Panel 2: Scatter plot nA vs nB
    ax2 = axes[1]
    colors = df['tile_seq']
    scatter = ax2.scatter(df['nA'], df['nB'], c=colors, cmap='viridis', alpha=0.7, s=50)
    ax2.plot([0, df['nA'].max() + 1], [0, df['nA'].max() + 1], 'k--', alpha=0.5, label='A=B')
    
    # Markeer pure stereo punt (1,1)
    ax2.scatter([1], [1], c='purple', s=200, marker='*', label='Pure stereo (1,1)')
    
    ax2.set_xlabel('nA')
    ax2.set_ylabel('nB')
    ax2.set_title('Sensor Balance')
    ax2.legend(loc='upper left', fontsize=8)
    plt.colorbar(scatter, ax=ax2, label='Tile sequence')
    
    # Panel 3: Balance over tijd
    ax3 = axes[2]
    balance = df['nA'] - df['nB']
    colors = ['blue' if b > 0 else 'red' if b < 0 else 'gray' for b in balance]
    ax3.bar(x, balance, color=colors, alpha=0.7, width=1.0)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    total_balance = df['nA'].sum() - df['nB'].sum()
    ax3.set_title(f'Sensor Balance (A-B) — Totaal: {total_balance:+.0f}')
    ax3.set_xlabel('Tile sequence')
    ax3.set_ylabel('nA - nB')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path / 'phase_distribution.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ phase_distribution.png")
    if show:
        plt.show()
    plt.close()


def plot_awareness_timeline(df: pd.DataFrame, output_path: Path = None, show: bool = False):
    """
    Plot 7: Awareness confidence tijdlijn
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    x = df['t_s']
    
    # Panel 1: Awareness components
    ax1 = axes[0]
    ax1.fill_between(x, 0, df['awareness_conf'], alpha=0.3, color='purple', label='Awareness')
    ax1.plot(x, df['awareness_conf'], 'purple', linewidth=2)
    
    if 'motion_conf' in df.columns:
        ax1.plot(x, df['motion_conf'], 'orange', linewidth=1.5, alpha=0.7, 
                linestyle='--', label='Motion conf')
    
    if 'direction_locked_conf' in df.columns:
        ax1.plot(x, df['direction_locked_conf'], 'green', linewidth=1.5, alpha=0.7,
                linestyle=':', label='Lock conf')
    
    ax1.set_ylabel('Confidence')
    ax1.set_title('Awareness Components')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_ylim(0, 1.1)
    
    # Panel 2: State timeline
    ax2 = axes[1]
    
    # Rotor state achtergrond
    rotor_colors = {'STILL': '#e0e0e0', 'MOVEMENT': '#d0f0d0'}
    if 'rotor_state' in df.columns:
        prev_state = None
        prev_t = x.iloc[0]
        for idx, row in df.iterrows():
            if row['rotor_state'] != prev_state and prev_state is not None:
                color = rotor_colors.get(prev_state, 'white')
                ax2.axvspan(prev_t, row['t_s'], alpha=0.5, color=color)
                prev_t = row['t_s']
            prev_state = row['rotor_state']
        if prev_state:
            color = rotor_colors.get(prev_state, 'white')
            ax2.axvspan(prev_t, x.iloc[-1], alpha=0.5, color=color)
    
    # Lock state lijn
    lock_map = {'UNLOCKED': 0, 'SOFT_LOCK': 0.5, 'LOCKED': 1}
    if 'direction_lock_state' in df.columns:
        lock_num = df['direction_lock_state'].map(lock_map).fillna(0)
        ax2.step(x, lock_num, where='post', color='green', linewidth=3, label='Lock state')
    
    ax2.set_xlabel('Tijd (s)')
    ax2.set_ylabel('State')
    ax2.set_yticks([0, 0.5, 1])
    ax2.set_yticklabels(['UNLOCKED', 'SOFT_LOCK', 'LOCKED'])
    ax2.set_title('State Timeline (achtergrond = RotorState)')
    ax2.legend(loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path / 'awareness_timeline.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ awareness_timeline.png")
    if show:
        plt.show()
    plt.close()


def generate_summary(df: pd.DataFrame, csv_path: str) -> str:
    """Genereer tekstuele samenvatting."""
    summary = []
    summary.append("=" * 60)
    summary.append(f"SYMPHONIA S02 v1.9 ANALYSE RAPPORT")
    summary.append(f"Bestand: {csv_path}")
    summary.append("=" * 60)
    summary.append("")
    
    # Basis stats
    summary.append("STATISTIEKEN:")
    summary.append(f"  Tiles: {len(df)}")
    summary.append(f"  Duur: {df['t_s'].iloc[-1]:.2f}s")
    summary.append(f"  Totaal cycles: {df['cycles_physical'].sum():.1f}")
    
    # Sensor balance
    total_A = df['nA'].sum()
    total_B = df['nB'].sum()
    summary.append(f"  Sensor A: {total_A:.0f}, B: {total_B:.0f}, balance: {total_A - total_B:+.0f}")
    
    # Pure stereo
    if 'is_pure_stereo' in df.columns:
        stereo = df['is_pure_stereo'].sum()
        summary.append(f"  Pure stereo tiles: {stereo} ({100*stereo/len(df):.0f}%)")
    
    summary.append("")
    
    # Resultaten
    summary.append("RESULTATEN:")
    if 'rotor_state' in df.columns:
        summary.append(f"  Final RotorState: {df['rotor_state'].iloc[-1]}")
    if 'direction_lock_state' in df.columns:
        summary.append(f"  Final LockState: {df['direction_lock_state'].iloc[-1]}")
    if 'direction_global_effective' in df.columns:
        summary.append(f"  Direction: {df['direction_global_effective'].iloc[-1]}")
    if 'rotations' in df.columns:
        summary.append(f"  Rotaties: {df['rotations'].iloc[-1]:.2f}")
    if 'compass_global_score' in df.columns:
        summary.append(f"  Max compass score: {df['compass_global_score'].abs().max():.4f}")
    
    summary.append("")
    
    # State transitions
    summary.append("STATE TRANSITIONS:")
    if 'rotor_state' in df.columns:
        prev = None
        for _, row in df.iterrows():
            curr = row['rotor_state']
            if curr != prev:
                summary.append(f"  Tile {row['tile_index']}: RotorState → {curr}")
            prev = curr
    
    if 'direction_lock_state' in df.columns:
        prev = None
        for _, row in df.iterrows():
            curr = row['direction_lock_state']
            if curr != prev:
                summary.append(f"  Tile {row['tile_index']}: LockState → {curr}")
            prev = curr
    
    summary.append("")
    summary.append("=" * 60)
    
    return "\n".join(summary)


def main():
    parser = argparse.ArgumentParser(
        description='Visualisatie toolkit voor Symphonia S02 v1.9 bench data'
    )
    parser.add_argument('csv_file', help='Input CSV bestand')
    parser.add_argument('--output-dir', '-o', help='Output directory voor plots')
    parser.add_argument('--show', '-s', action='store_true', help='Toon plots interactief')
    parser.add_argument('--summary', action='store_true', help='Print alleen tekstuele samenvatting')
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"❌ Bestand niet gevonden: {csv_path}")
        return 1
    
    print(f"[i] Laden: {csv_path}")
    df = load_data(str(csv_path))
    print(f"[i] {len(df)} tiles geladen")
    
    # Print summary
    summary = generate_summary(df, str(csv_path))
    print(summary)
    
    if args.summary:
        return 0
    
    # Output directory
    output_path = None
    if args.output_dir:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"\n[i] Output directory: {output_path}")
    
    # Genereer plots
    print("\n[i] Genereren plots...")
    
    plot_overview_dashboard(df, output_path, args.show)
    plot_state_machine(df, output_path, args.show)
    plot_cycles_and_rotations(df, output_path, args.show)
    plot_compass_and_lock(df, output_path, args.show)
    plot_rpm_analysis(df, output_path, args.show)
    plot_phase_distribution(df, output_path, args.show)
    plot_awareness_timeline(df, output_path, args.show)
    
    print("\n✅ Klaar!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
