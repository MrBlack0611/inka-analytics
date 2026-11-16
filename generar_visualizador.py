import argparse
import json
import os
import xml.etree.ElementTree as ET
import pandas as pd

# Argument parser
parser = argparse.ArgumentParser(description="Generador de Visualizador 2D por Minutos de Partido")
parser.add_argument("--minuto", type=float, default=0.0, help="Minuto del partido (ej. 0 para el inicio, 4 para minuto 4, 50 para segundo tiempo)")
parser.add_argument("--duracion", type=int, default=60, help="Duracion en segundos a visualizar (por defecto 60 segundos)")
parser.add_argument("--sub", type=str, default="evaluation/results/inka_analytics_v11.csv", help="Archivo de predicciones a comparar con la realidad (por defecto inka_analytics_v11.csv)")

args = parser.parse_args()

# Constants
FPS = 25
KICKOFF_1ST_HALF = 11185   # Frame exacto del saque inicial 1er tiempo
KICKOFF_2ND_HALF = 113742  # Frame exacto del saque inicial 2do tiempo

# Calcular el frame de inicio segun el minuto del partido
if args.minuto < 45.0:
    start_frame = int(KICKOFF_1ST_HALF + (args.minuto * 60 * FPS))
    half_name = "1er Tiempo"
    match_min_display = args.minuto
else:
    start_frame = int(KICKOFF_2ND_HALF + ((args.minuto - 45.0) * 60 * FPS))
    half_name = "2do Tiempo"
    match_min_display = args.minuto

total_frames = args.duracion * FPS
end_frame = start_frame + total_frames

print(f"--- Configuración del Visualizador ---")
print(f"Minuto del partido: {args.minuto} ({half_name})")
print(f"Duración: {args.duracion} segundos ({total_frames} frames)")
print(f"Rango de frames: {start_frame} a {end_frame}")
print(f"Leyendo tracking y metadatos...")

# Cargar metadatos de equipos
meta_path = 'data/tactical/BAR-ATH_tactical_atd.xml'
root = ET.parse(meta_path).getroot()
teams = list(root.iter('Team'))
home_id = teams[0].get('id')
home_name = teams[0].findtext('Name')
away_name = teams[1].findtext('Name')

slots = [p.get('teamId') for p in root.iter('Player')]
team_of_slot = [0 if t == home_id else 1 for t in slots]

# Cargar GT oficial y predicciones del modelo a comparar
gt_path = 'evaluation/gt/BAR-ATH_possession_gt_public.csv'
gt = pd.read_csv(gt_path)

if os.path.exists(args.sub):
    pred_df = pd.read_csv(args.sub)
    pred_name = os.path.basename(args.sub)
else:
    pred_df = None
    pred_name = "Sin prediccion cargada"

frames_data = []

with open('data/tactical/BAR-ATH_tactical_atd.txt') as f:
    for idx, line in enumerate(f):
        if idx < start_frame:
            continue
        if idx >= end_frame:
            break
        parts = line.strip().split(':')
        if len(parts) != 3:
            continue
        f_num = int(parts[0])
        players_raw = parts[1].split(';')
        ball_raw = parts[2].split(',')
        
        # Posicion balon
        if ball_raw[0] != 'NaN':
            ball = {'x': round(float(ball_raw[0]) * 105, 2), 'y': round(float(ball_raw[1]) * 68, 2)}
        else:
            ball = None
            
        # Posicion jugadores
        plist = []
        for s_idx, p_str in enumerate(players_raw):
            coords = p_str.split(',')
            if coords[0] != 'NaN':
                plist.append({
                    'x': round(float(coords[0]) * 105, 2),
                    'y': round(float(coords[1]) * 68, 2),
                    't': team_of_slot[s_idx]
                })
        
        gt_label = str(gt.iloc[f_num]['label']) if f_num < len(gt) else 'X'
        pred_label = str(pred_df.iloc[f_num]['label']) if pred_df is not None and f_num < len(pred_df) else 'N/A'
        
        frames_data.append({
            'f': f_num,
            'b': ball,
            'p': plist,
            'gt': gt_label,
            'pred': pred_label
        })

frames_json = json.dumps(frames_data)

html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Visualizador Táctico 2D - Minuto {args.minuto}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #10141a;
            color: #e0e6ed;
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        h1 {{ margin-bottom: 5px; color: #fff; font-size: 22px; }}
        .subtitle {{ color: #8fa0b5; margin-bottom: 15px; font-size: 14px; }}
        .badge {{
            background: #0066cc;
            color: #fff;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            margin-left: 6px;
        }}
        .canvas-container {{
            position: relative;
            background: #236c34;
            border: 4px solid #3a8a4f;
            border-radius: 8px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.6);
        }}
        canvas {{ display: block; }}
        .controls {{
            margin-top: 15px;
            display: flex;
            align-items: center;
            gap: 15px;
            background: #1a222d;
            padding: 12px 24px;
            border-radius: 8px;
            width: 920px;
            box-sizing: border-box;
            border: 1px solid #2d3848;
        }}
        button {{
            background: #0066cc;
            color: white;
            border: none;
            padding: 8px 18px;
            font-size: 15px;
            font-weight: bold;
            border-radius: 6px;
            cursor: pointer;
            transition: 0.2s;
        }}
        button:hover {{ background: #0052a3; }}
        input[type=range] {{
            flex-grow: 1;
            cursor: pointer;
        }}
        .hud {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            width: 920px;
            margin-top: 15px;
        }}
        .card {{
            background: #1a222d;
            padding: 12px 16px;
            border-radius: 8px;
            border: 1px solid #2d3848;
            border-left: 4px solid #0066cc;
        }}
        .card.alert {{
            border-left-color: #e63946;
            background: #2c161d;
        }}
        .card.success {{
            border-left-color: #2a9d8f;
        }}
        .card-label {{ font-size: 11px; color: #8fa0b5; text-transform: uppercase; font-weight: 600; }}
        .card-val {{ font-size: 16px; font-weight: bold; margin-top: 4px; }}
        .legend {{
            display: flex;
            gap: 25px;
            margin-top: 12px;
            font-size: 14px;
        }}
        .legend-item {{ display: flex; align-items: center; gap: 8px; }}
        .dot {{ width: 14px; height: 14px; border-radius: 50%; display: inline-block; }}
    </style>
</head>
<body>
    <h1>⚽ Visualizador 2D: Minuto {args.minuto} <span class="badge">{half_name}</span></h1>
    <div class="subtitle">FC Barcelona vs Athletic Club | Comparando modelo con Ground Truth</div>

    <div class="canvas-container">
        <canvas id="pitch" width="920" height="596"></canvas>
    </div>

    <div class="controls">
        <button id="playBtn">Play</button>
        <input type="range" id="slider" min="0" max="{len(frames_data) - 1}" value="0">
        <span id="frameCounter" style="font-family: monospace; font-size: 15px;">Frame: {start_frame}</span>
    </div>

    <div class="legend">
        <div class="legend-item"><span class="dot" style="background: #1a3b8b; border: 2px solid #edbb00;"></span> <b>Barcelona (Local / H)</b></div>
        <div class="legend-item"><span class="dot" style="background: #e63946; border: 2px solid #ffffff;"></span> <b>Athletic Club (Visitante / A)</b></div>
        <div class="legend-item"><span class="dot" style="background: #ffff33; border: 2px solid #000000;"></span> <b>Balón</b></div>
    </div>

    <div class="hud">
        <div class="card">
            <div class="card-label">Reloj de Partido</div>
            <div class="card-val" id="timeVal">00:00</div>
        </div>
        <div class="card">
            <div class="card-label">Balón en Cancha</div>
            <div class="card-val" id="ballVal">Rastreando</div>
        </div>
        <div class="card success">
            <div class="card-label">Realidad (Ground Truth)</div>
            <div class="card-val" id="gtVal">...</div>
        </div>
        <div class="card" id="predCard">
            <div class="card-label">Predicción ({pred_name})</div>
            <div class="card-val" id="predVal">...</div>
        </div>
    </div>

    <script>
        const data = {frames_json};
        const startFrame = {start_frame};
        const baseMin = {args.minuto};
        const canvas = document.getElementById('pitch');
        const ctx = canvas.getContext('2d');
        const slider = document.getElementById('slider');
        const playBtn = document.getElementById('playBtn');
        const frameCounter = document.getElementById('frameCounter');
        const timeVal = document.getElementById('timeVal');
        const ballVal = document.getElementById('ballVal');
        const gtVal = document.getElementById('gtVal');
        const predVal = document.getElementById('predVal');
        const predCard = document.getElementById('predCard');

        const PITCH_W = 105;
        const PITCH_H = 68;
        const MARGIN = 30;
        const W = canvas.width - 2 * MARGIN;
        const H = canvas.height - 2 * MARGIN;

        function toX(x_m) {{ return MARGIN + (x_m / PITCH_W) * W; }}
        function toY(y_m) {{ return MARGIN + (y_m / PITCH_H) * H; }}

        function drawPitch() {{
            ctx.fillStyle = '#236c34';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Franjas de corte de pasto
            ctx.fillStyle = 'rgba(255, 255, 255, 0.03)';
            for (let i = 0; i < 10; i++) {{
                if (i % 2 === 0) {{
                    ctx.fillRect(MARGIN + i * (W / 10), MARGIN, W / 10, H);
                }}
            }}

            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;

            // Lineas exteriores
            ctx.strokeRect(MARGIN, MARGIN, W, H);

            // Linea media
            ctx.beginPath();
            ctx.moveTo(toX(52.5), MARGIN);
            ctx.lineTo(toX(52.5), MARGIN + H);
            ctx.stroke();

            // Circulo central
            ctx.beginPath();
            ctx.arc(toX(52.5), toY(34), (9.15 / PITCH_W) * W, 0, Math.PI * 2);
            ctx.stroke();

            // Punto central
            ctx.fillStyle = '#ffffff';
            ctx.beginPath();
            ctx.arc(toX(52.5), toY(34), 3, 0, Math.PI * 2);
            ctx.fill();

            // Areas penal
            ctx.strokeRect(toX(0), toY(13.85), (16.5 / PITCH_W) * W, (40.3 / PITCH_H) * H);
            ctx.strokeRect(toX(0), toY(24.85), (5.5 / PITCH_W) * W, (18.3 / PITCH_H) * H);

            ctx.strokeRect(toX(88.5), toY(13.85), (16.5 / PITCH_W) * W, (40.3 / PITCH_H) * H);
            ctx.strokeRect(toX(99.5), toY(24.85), (5.5 / PITCH_W) * W, (18.3 / PITCH_H) * H);
        }}

        let curIdx = 0;
        let isPlaying = false;
        let timer = null;

        function renderFrame(idx) {{
            const f = data[idx];
            drawPitch();

            // Dibujar jugadores
            for (const p of f.p) {{
                const px = toX(p.x);
                const py = toY(p.y);
                ctx.beginPath();
                ctx.arc(px, py, 7, 0, Math.PI * 2);
                if (p.t === 0) {{
                    ctx.fillStyle = '#1a3b8b';
                    ctx.fill();
                    ctx.strokeStyle = '#edbb00';
                    ctx.lineWidth = 2;
                    ctx.stroke();
                }} else {{
                    ctx.fillStyle = '#e63946';
                    ctx.fill();
                    ctx.strokeStyle = '#ffffff';
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                }}
            }}

            // Dibujar balon
            if (f.b) {{
                const bx = toX(f.b.x);
                const by = toY(f.b.y);
                ctx.beginPath();
                ctx.arc(bx, by, 6, 0, Math.PI * 2);
                ctx.fillStyle = '#ffff33';
                ctx.fill();
                ctx.strokeStyle = '#000000';
                ctx.lineWidth = 1.5;
                ctx.stroke();
                ballVal.innerHTML = 'Pos: (' + f.b.x + 'm, ' + f.b.y + 'm)';
                ballVal.style.color = '#70e000';
            }} else {{
                ballVal.innerHTML = '⚠️ PERDIDO (NaN)';
                ballVal.style.color = '#ff9f1c';
            }}

            frameCounter.innerText = 'Frame: ' + f.f;
            slider.value = idx;
            
            // Reloj partido
            const secElapsed = idx / 25;
            const totalSec = baseMin * 60 + secElapsed;
            const minDisp = Math.floor(totalSec / 60);
            const secDisp = Math.floor(totalSec % 60);
            timeVal.innerText = minDisp + ':' + (secDisp < 10 ? '0' : '') + secDisp;

            const mapName = {{ 'H': 'H (Barcelona)', 'A': 'A (Athletic)', 'D': 'D (Balón Parado)', 'X': 'X (Pausa / Privado)' }};
            gtVal.innerText = mapName[f.gt] || f.gt;
            predVal.innerText = mapName[f.pred] || f.pred;

            if (f.gt !== 'X' && f.gt !== f.pred) {{
                predCard.className = 'card alert';
                predVal.innerHTML += ' <span style="font-size: 13px;">❌ [ERROR]</span>';
            }} else if (f.gt !== 'X') {{
                predCard.className = 'card success';
                predVal.innerHTML += ' <span style="font-size: 13px;">✅ [OK]</span>';
            }} else {{
                predCard.className = 'card';
            }}
        }}

        slider.addEventListener('input', (e) => {{
            curIdx = parseInt(e.target.value);
            renderFrame(curIdx);
        }});

        playBtn.addEventListener('click', () => {{
            isPlaying = !isPlaying;
            playBtn.innerText = isPlaying ? 'Pausa' : 'Play';
            if (isPlaying) {{
                timer = setInterval(() => {{
                    curIdx++;
                    if (curIdx >= data.length) {{
                        curIdx = 0;
                    }}
                    renderFrame(curIdx);
                }}, 40); // 25 fps = 40 ms
            }} else {{
                clearInterval(timer);
            }}
        }});

        renderFrame(0);
    </script>
</body>
</html>
"""

output_path = os.path.abspath('visualizador_cancha.html')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"-> Archivo generado con éxito en: {output_path}")
