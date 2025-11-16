# -*- coding: utf-8 -*-
"""
Flask 웹 서버 - 감정 분석 및 색상 추천 API
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sys
import os
import json
import time

# our_model 폴더의 improved_emotion_model.py에서 함수 임포트
sys.path.append(os.path.join(os.path.dirname(__file__), 'our_model'))

try:
    from improved_emotion_model import analyze_emotion_and_color
    AI_MODEL_AVAILABLE = True
    print("🤖 개선된 AI 모델 로딩 성공!")
except Exception as e:
    print(f"❌ 개선된 AI 모델 로딩 실패: {e}")
    # 폴백: 기존 simple_emotion_model 사용
    try:
        from simple_emotion_model import analyze_emotion_and_color as fallback_analyze
        analyze_emotion_and_color = fallback_analyze
        AI_MODEL_AVAILABLE = True
        print("🔄 폴백 모델 사용")
    except Exception as e2:
        print(f"❌ 폴백 모델도 실패: {e2}")
        AI_MODEL_AVAILABLE = False

app = Flask(__name__)
CORS(app)  # CORS 설정 - 다른 도메인에서의 요청 허용

# 전역 변수: 학습된 모델과 데이터 저장
loaded_models = None

@app.route('/')
def index():
    """메인 웹페이지 렌더링"""
    return render_template('index.html')

@app.route('/diary/<int:diary_id>')
def diary_detail(diary_id):
    """전체화면 일기 상세 페이지"""
    return render_template('diary_viewer.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_emotion():
    """감정 분석 및 색상 추천 API"""
    try:
        data = request.get_json()
        diary_entry = data.get('text', '')
        
        if not diary_entry.strip():
            return jsonify({'error': '일기 내용이 비어있습니다.'}), 400
        
        if not AI_MODEL_AVAILABLE:
            # AI 모델이 없는 경우 기본값 반환
            return jsonify({
                'success': True,
                'emotion': 'Neutral',
                'color_hex': '#667eea',
                'color_name': '파란색',
                'tone': '기본 톤'
            })
        
        # 감정 분석 실행
        result = analyze_emotion_and_color(diary_entry, show_visualization=False)
        
        return jsonify({
            'success': True,
            'emotion': result['emotion'],
            'color_hex': result['color_hex'],
            'color_name': result['color_name'],
            'tone': result['tone']
        })
        
    except Exception as e:
        print(f"감정 분석 중 오류 발생: {e}")
        return jsonify({'error': f'감정 분석 중 오류가 발생했습니다: {str(e)}'}), 500

@app.route('/api/save-diary', methods=['POST'])
def save_diary():
    """일기 저장 API"""
    try:
        data = request.get_json()
        
        diary_data = {
            'id': int(time.time() * 1000),  # 고유 ID
            'date': data.get('date'),
            'title': data.get('title'),
            'content': data.get('content'),
            'mood': data.get('mood', ''),  # 사용자가 선택한 기분 (기본값: 빈 문자열)
            'emotion': data.get('emotion', ''),  # AI가 분석한 감정 (기본값: 빈 문자열)
            'color_hex': data.get('color_hex', '#667eea'),  # 추천 색상 (기본값: 파란색)
            'color_name': data.get('color_name', '파란색'),
            'tone': data.get('tone', '기본 톤'),
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # JSON 파일에 저장 (실제 서비스에서는 DB 사용)
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
            'diary_id': diary_data['id']
        })
        
    except Exception as e:
        print(f"일기 저장 중 오류 발생: {e}")
        return jsonify({'error': f'일기 저장 중 오류가 발생했습니다: {str(e)}'}), 500

@app.route('/api/diaries', methods=['GET'])
def get_diaries():
    """저장된 일기 목록 반환 API"""
    try:
        diary_file = 'diaries.json'
        
        if not os.path.exists(diary_file):
            return jsonify({'success': True, 'diaries': []})
        
        with open(diary_file, 'r', encoding='utf-8') as f:
            diaries = json.load(f)
        
        # 최신순으로 정렬
        diaries.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return jsonify({
            'success': True,
            'diaries': diaries
        })
        
    except Exception as e:
        print(f"일기 목록 조회 중 오류 발생: {e}")
        return jsonify({'error': f'일기 목록 조회 중 오류가 발생했습니다: {str(e)}'}), 500

@app.route('/api/diary/<int:diary_id>', methods=['DELETE'])
def delete_diary(diary_id):
    """일기 삭제 API"""
    try:
        diary_file = 'diaries.json'
        
        if not os.path.exists(diary_file):
            return jsonify({'error': '삭제할 일기를 찾을 수 없습니다.'}), 404
        
        with open(diary_file, 'r', encoding='utf-8') as f:
            diaries = json.load(f)
        
        # ID로 일기 찾기 및 삭제
        original_count = len(diaries)
        diaries = [d for d in diaries if d.get('id') != diary_id]
        
        if len(diaries) == original_count:
            return jsonify({'error': '삭제할 일기를 찾을 수 없습니다.'}), 404
        
        with open(diary_file, 'w', encoding='utf-8') as f:
            json.dump(diaries, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'success': True,
            'message': '일기가 성공적으로 삭제되었습니다.'
        })
        
    except Exception as e:
        print(f"일기 삭제 중 오류 발생: {e}")
        return jsonify({'error': f'일기 삭제 중 오류가 발생했습니다: {str(e)}'}), 500

@app.route('/api/diary/<int:diary_id>', methods=['GET'])
def get_diary(diary_id):
    """개별 일기 조회 API"""
    try:
        diary_file = 'diaries.json'
        
        if not os.path.exists(diary_file):
            return jsonify({'error': '일기를 찾을 수 없습니다.'}), 404
        
        with open(diary_file, 'r', encoding='utf-8') as f:
            diaries = json.load(f)
        
        # ID로 일기 찾기
        diary = next((d for d in diaries if d.get('id') == diary_id), None)
        
        if not diary:
            return jsonify({'error': '일기를 찾을 수 없습니다.'}), 404
        
        return jsonify({
            'success': True,
            'diary': diary
        })
        
    except Exception as e:
        print(f"일기 조회 중 오류 발생: {e}")
        return jsonify({'error': f'일기 조회 중 오류가 발생했습니다: {str(e)}'}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 일기장 웹 서버 시작!")
    print("📍 주소: http://localhost:8080")
    print("📋 API 엔드포인트:")
    print("   - POST /api/analyze          : 감정 분석 및 색상 추천")
    print("   - POST /api/save-diary       : 일기 저장")
    print("   - GET  /api/diaries          : 일기 목록 조회")
    print("   - DELETE /api/diary/<id>     : 일기 삭제")
    print(f"🤖 AI 모델 상태: {'활성화' if AI_MODEL_AVAILABLE else '비활성화'}")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)
