#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
삭제 API 테스트 스크립트
"""
import requests
import json
import time

# 서버 URL
BASE_URL = 'http://localhost:5001'

def test_delete():
    """삭제 기능 테스트"""
    
    print("=" * 60)
    print("🧪 삭제 API 테스트")
    print("=" * 60)
    
    # 1. 일기 목록 조회
    print("\n1️⃣ 일기 목록 조회 중...")
    try:
        response = requests.get(f'{BASE_URL}/api/diaries')
        diaries = response.json()
        print(f"✅ 조회 성공! 총 {len(diaries)}개 일기")
        
        if not diaries:
            print("❌ 삭제할 일기가 없습니다.")
            return
        
        # 첫 번째 일기 정보
        first_diary = diaries[0]
        diary_id = first_diary['id']
        print(f"\n삭제 대상:")
        print(f"  - ID: {diary_id} (타입: {type(diary_id).__name__})")
        print(f"  - 제목: {first_diary['title']}")
        print(f"  - 내용: {first_diary['content'][:50]}...")
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        return
    
    # 2. 삭제 요청
    print(f"\n2️⃣ 일기 삭제 중 (ID: {diary_id})...")
    try:
        url = f'{BASE_URL}/api/diary/{diary_id}'
        print(f"   요청 URL: {url}")
        print(f"   요청 방식: DELETE")
        
        response = requests.delete(url)
        print(f"   응답 상태: {response.status_code}")
        print(f"   응답 내용: {response.text}")
        
        if response.status_code == 200:
            print("✅ 삭제 성공!")
            result = response.json()
            print(f"   메시지: {result.get('message', 'N/A')}")
        else:
            print(f"❌ 삭제 실패! (상태 코드: {response.status_code})")
            print(f"   오류: {response.json().get('error', 'N/A')}")
            return
            
    except Exception as e:
        print(f"❌ 오류: {e}")
        return
    
    # 3. 삭제 확인
    print(f"\n3️⃣ 삭제 확인 중...")
    try:
        response = requests.get(f'{BASE_URL}/api/diaries')
        diaries_after = response.json()
        print(f"✅ 확인 성공! 남은 일기: {len(diaries_after)}개")
        
        if len(diaries_after) < len(diaries):
            print("✅ 일기가 정상적으로 삭제되었습니다!")
        else:
            print("❌ 일기가 삭제되지 않았습니다!")
            
    except Exception as e:
        print(f"❌ 오류: {e}")

if __name__ == '__main__':
    print("⏳ 3초 후 테스트 시작...\n")
    time.sleep(3)
    test_delete()
