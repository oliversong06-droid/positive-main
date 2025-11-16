# -*- coding: utf-8 -*-
"""
간단한 Flask 웹 서버 - 일기장 (AI 모델 포함)
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import time
import os
import sys

# our_model 폴더 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'our_model'))

try:
    from simple_emotion_model import analyze_emotion_and_color
    AI_MODEL_LOADED = True
    print("✅ AI 감정 분석 모델 로드 성공!")
except Exception as e:
    print(f"⚠️ AI 모델 로드 실패: {e}")
    AI_MODEL_LOADED = False

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/read-diary')
def read_diary():
    return render_template('read-diary.html')

@app.route('/api/save-diary', methods=['POST'])
def save_diary():
    try:
        data = request.get_json()
        
        emotion = ''
        color_hex = '#667eea'
        color_name = '파란색'
        tone = ''
        
        if AI_MODEL_LOADED:
            try:
                text_to_analyze = f"{data.get('title', '')} {data.get('content', '')}"
                analysis_result = analyze_emotion_and_color(text_to_analyze)
                
                emotion = analysis_result.get('emotion', '')
                color_hex = analysis_result.get('color_hex', '#667eea')
                color_name = analysis_result.get('color_name', '파란색')
                tone = analysis_result.get('tone', '')
                
                print(f"✅ 감정 분석 완료: {emotion}")
            except Exception as e:
                print(f"⚠️ 감정 분석 중 오류: {e}")
        
        diary_data = {
            'id': int(time.time() * 1000),
            'date': data.get('date'),
            'title': data.get('title'),
            'content': data.get('content'),
            'emotion': emotion,
            'color_hex': color_hex,
            'color_name': color_name,
            'tone': tone,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        diary_file = 'diaries.json'
        diaries = []
        
        if os.path.exists(diary_file):
            with open(diary_file, 'r', encoding='utf-8') as f:
                diaries = json.load(f)
        
        diaries.append(diary_data)
        
        with open(diary_file, 'w', encoding='utf-8') as f:
            json.dump(diaries, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'success': True,
            'message': '일기가 성공적으로 저장되었습니다.',
            'diary_id': diary_data['id'],
            'emotion': emotion,
            'color_hex': color_hex,
            'color_name': color_name
        })
        
    except Exception as e:
        print(f"일기 저장 중 오류 발생: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/diaries', methods=['GET'])
def get_diaries():
    try:
        diary_file = 'diaries.json'
        
        if not os.path.exists(diary_file):
            return jsonify([])
        
        with open(diary_file, 'r', encoding='utf-8') as f:
            diaries = json.load(f)
        
        diaries.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return jsonify(diaries)
        
    except Exception as e:
        print(f"일기 목록 조회 중 오류 발생: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete-diary', methods=['POST'])
def delete_diary():
    try:
        data = request.get_json()
        diary_id = data.get('id')
        
        print(f"\n{'='*60}")
        print(f"🗑️ 삭제 요청: ID={diary_id}")
        print(f"{'='*60}")
        
        if not diary_id:
            return jsonify({'error': 'ID가 필요합니다.'}), 400
        
        try:
            diary_id = int(diary_id)
        except (ValueError, TypeError):
            return jsonify({'error': '유효하지 않은 일기 ID입니다.'}), 400
        
        diary_file = 'diaries.json'
        
        if not os.path.exists(diary_file):
            return jsonify({'error': '삭제할 일기를 찾을 수 없습니다.'}), 404
        
        with open(diary_file, 'r', encoding='utf-8') as f:
            diaries = json.load(f)
        
        original_count = len(diaries)
        print(f"📝 현재 일기: {original_count}개")
        print(f"📝 현재 ID들: {[d.get('id') for d in diaries]}")
        
        diaries = [d for d in diaries if d.get('id') != diary_id]
        
        if len(diaries) == original_count:
            print(f"❌ 일기를 찾을 수 없음: ID={diary_id}")
            return jsonify({'error': '삭제할 일기를 찾을 수 없습니다.'}), 404
        
        with open(diary_file, 'w', encoding='utf-8') as f:
            json.dump(diaries, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 일기 삭제 성공: {original_count} -> {len(diaries)}")
        
        return jsonify({
            'success': True,
            'message': '일기가 성공적으로 삭제되었습니다.'
        })
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 일기장 웹 서버 시작!")
    print("📍 http://localhost:5001")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5001, debug=True)
