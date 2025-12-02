#!/usr/bin/env python3
"""Generate HTML report with embedded images."""

import json
import base64
from pathlib import Path

def image_to_base64(image_path):
    """Convert image to base64 data URI."""
    try:
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{data}"
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return ""

def main():
    # Paths
    data_root = "/home/tim/projects/vlm-agentic-attack/data/mssbench"
    annotations_path = Path(__file__).parent / "results" / "chat-10-each_annotations.json"
    output_path = Path(__file__).parent / "results" / "annotation_summary.html"
    
    # Load annotations
    with open(annotations_path, "r") as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples")
    
    # Generate HTML
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>MSSBench Annotation Summary</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #fff; }
        h1, h2 { color: #333; }
        table { border-collapse: collapse; margin: 15px 0; }
        th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
        th { background: #f0f0f0; }
        .sample { border: 1px solid #ddd; padding: 15px; margin: 15px 0; background: #fafafa; }
        .sample-header { font-weight: bold; font-size: 16px; margin-bottom: 10px; }
        .sample img { max-width: 300px; display: block; margin: 10px 0; }
        .query { background: #e8f4fc; padding: 8px; margin: 5px 0; }
        .result-table { width: 100%; margin-top: 10px; }
        .result-table td { vertical-align: top; }
        .unsafe { color: #c00; font-weight: bold; }
        .safe { color: #080; }
        .ambiguous { color: #888; }
        .neutral { color: #666; }
        .tag { display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 12px; margin-left: 10px; }
        .tag-high { background: #28a745; color: white; }
        .tag-mod { background: #ffc107; color: black; }
        .tag-comp { background: #17a2b8; color: white; }
        .stats-box { background: #f5f5f5; padding: 15px; margin: 20px 0; }
    </style>
</head>
<body>
    <h1>MSSBench Dual-LLM Annotation Summary</h1>
    <p><b>Models:</b> GPT-5 + Gemini 2.5 Pro | <b>Dataset:</b> chat-10-each.json (40 samples, 10 per type)</p>
    
    <div class="stats-box">
        <h2>Overall Statistics</h2>
        <table>
            <tr><th>Metric</th><th>Count</th><th>Rate</th></tr>
            <tr><td>High-confidence Unsafe (both agree)</td><td>5</td><td>12.5%</td></tr>
            <tr><td>Moderate-confidence Unsafe (1 says unsafe)</td><td>9</td><td>22.5%</td></tr>
            <tr><td>Ambiguous</td><td>3</td><td>7.5%</td></tr>
            <tr><td>Safe (both agree safe)</td><td>23</td><td>57.5%</td></tr>
            <tr style="background:#ffe;"><td><b>Total Unsafe Rate</b></td><td><b>14</b></td><td><b>35%</b></td></tr>
            <tr style="background:#e8f4fc;"><td><b>Compositional Unsafe</b></td><td><b>9</b></td><td><b>22.5%</b></td></tr>
        </table>
        
        <h3>By Type</h3>
        <table>
            <tr><th>Type</th><th>Total</th><th>Unsafe</th><th>Unsafe Rate</th><th>Compositional</th></tr>
            <tr><td>illegal</td><td>10</td><td>7</td><td><b>70%</b></td><td>6</td></tr>
            <tr><td>harmful</td><td>10</td><td>6</td><td>60%</td><td>2</td></tr>
            <tr><td>property</td><td>10</td><td>1</td><td>10%</td><td>1</td></tr>
            <tr><td>offensive</td><td>10</td><td>0</td><td>0%</td><td>0</td></tr>
        </table>
        
        <h3>Key Findings</h3>
        <ul>
            <li><b>Only 35% of "unsafe" samples are actually unsafe</b> - MSSBench quality issue</li>
            <li><b>offensive type completely failed</b> (0%) - LLMs see these as etiquette issues, not safety risks</li>
            <li><b>illegal type is best</b> (70% unsafe, 60% compositional) - clear rule violations</li>
            <li><b>Compositional</b> = text safe + image safe → combined unsafe (the real research target)</li>
        </ul>
    </div>
    
    <h2>All 40 Samples</h2>
"""
    
    def fmt(val):
        if val == 'unsafe':
            return '<span class="unsafe">unsafe</span>'
        elif val == 'safe':
            return '<span class="safe">safe</span>'
        elif val == 'ambiguous':
            return '<span class="ambiguous">ambiguous</span>'
        elif val == 'neutral':
            return '<span class="neutral">neutral</span>'
        return val or 'N/A'
    
    for i, s in enumerate(data):
        idx = i + 1
        
        # Tags
        tags = ''
        if s.get('agreement') == 'high-confidence-unsafe':
            tags += '<span class="tag tag-high">HIGH</span>'
        if s.get('agreement') == 'moderate-confidence-unsafe':
            tags += '<span class="tag tag-mod">MOD</span>'
        if s.get('is_compositional'):
            tags += '<span class="tag tag-comp">COMP</span>'
        
        # Load image
        img_path = f"{data_root}/chat/{s['unsafe_image_path']}"
        img_data = image_to_base64(img_path)
        
        # GPT-5 results
        gpt5 = s.get('gpt5') or {}
        gemini = s.get('gemini') or {}
        
        html += f"""
    <div class="sample">
        <div class="sample-header">#{idx} | Type: {s['type']} {tags}</div>
        <img src="{img_data}" alt="{s['unsafe_image_path']}">
        <div class="query"><b>Query:</b> {s['query']}</div>
        <table class="result-table">
            <tr>
                <th width="50%">GPT-5</th>
                <th width="50%">Gemini</th>
            </tr>
            <tr>
                <td>
                    text: {fmt(gpt5.get('text_only'))} | image: {fmt(gpt5.get('image_only'))} | <b>combined: {fmt(gpt5.get('combined'))}</b><br>
                    <i>{gpt5.get('rationale', '')}</i>
                </td>
                <td>
                    text: {fmt(gemini.get('text_only'))} | image: {fmt(gemini.get('image_only'))} | <b>combined: {fmt(gemini.get('combined'))}</b><br>
                    <i>{gemini.get('rationale', '')}</i>
                </td>
            </tr>
        </table>
    </div>
"""
    
    html += """
</body>
</html>
"""
    
    # Write output
    with open(output_path, "w") as f:
        f.write(html)
    
    print(f"Generated: {output_path}")

if __name__ == "__main__":
    main()
