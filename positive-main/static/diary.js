// 전체화면 일기 페이지 JavaScript

class DiaryDetailApp {
    constructor() {
        this.init();
    }

    init() {
        // 테마색 설정
        this.setThemeColor();
    }

    async loadDiary(diaryId) {
        try {
            const response = await fetch(`/api/diary/${diaryId}`);
            const data = await response.json();

            if (data.success) {
                this.renderDiary(data.diary);
            } else {
                this.showError(data.error || '일기를 찾을 수 없습니다.');
            }
        } catch (error) {
            console.error('일기 로드 중 오류:', error);
            this.showError('일기를 불러오는 중 오류가 발생했습니다.');
        }
    }

    renderDiary(diary) {
        // 테마색 설정
        if (diary.color_hex) {
            const themeColorDark = this.darkenColor(diary.color_hex, 20);
            document.documentElement.style.setProperty('--theme-color', diary.color_hex);
            document.documentElement.style.setProperty('--theme-color-dark', themeColorDark);
        }

        // 일기 내용 표시
        document.getElementById('diaryTitle').textContent = diary.title;
        document.getElementById('diaryText').textContent = diary.content;
        document.getElementById('diaryDate').textContent = this.formatDate(diary.date);
        document.getElementById('diaryEmotion').textContent = diary.emotion ? `🤖 ${diary.emotion}` : '';
        document.getElementById('diaryMood').textContent = diary.mood || '기분 미선택';
        document.getElementById('diaryColor').textContent = diary.color_name || '기본 색상';
    }

    showError(message) {
        document.body.innerHTML = `
            <div style="
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                text-align: center;
                font-size: 1.2em;
            ">
                <div>
                    <h2>⚠️ 오류</h2>
                    <p>${message}</p>
                    <a href="/" style="
                        display: inline-block;
                        margin-top: 20px;
                        padding: 12px 24px;
                        background: rgba(255, 255, 255, 0.2);
                        color: white;
                        text-decoration: none;
                        border-radius: 8px;
                        border: 1px solid rgba(255, 255, 255, 0.3);
                    ">홈으로 돌아가기</a>
                </div>
            </div>
        `;
    }

    formatDate(dateString) {
        const date = new Date(dateString + 'T00:00:00');
        const options = { 
            year: 'numeric', 
            month: 'long', 
            day: 'numeric', 
            weekday: 'long' 
        };
        return date.toLocaleDateString('ko-KR', options);
    }

    // 색상을 어둡게 만드는 함수
    darkenColor(hexColor, percent) {
        // HEX 색상을 RGB로 변환
        const r = parseInt(hexColor.slice(1, 3), 16);
        const g = parseInt(hexColor.slice(3, 5), 16);
        const b = parseInt(hexColor.slice(5, 7), 16);
        
        // 어둡게 만들기 (percent만큼 감소)
        const newR = Math.max(0, Math.floor(r * (100 - percent) / 100));
        const newG = Math.max(0, Math.floor(g * (100 - percent) / 100));
        const newB = Math.max(0, Math.floor(b * (100 - percent) / 100));
        
        // 다시 HEX로 변환
        return '#' + 
            newR.toString(16).padStart(2, '0') + 
            newG.toString(16).padStart(2, '0') + 
            newB.toString(16).padStart(2, '0');
    }

    setThemeColor() {
        // 기본 테마색 설정
        document.documentElement.style.setProperty('--theme-color', '#667eea');
        document.documentElement.style.setProperty('--theme-color-dark', '#764ba2');
    }
}

// 앱 초기화
let diaryApp;
document.addEventListener('DOMContentLoaded', () => {
    diaryApp = new DiaryDetailApp();
});