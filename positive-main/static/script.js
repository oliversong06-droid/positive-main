// 일기 데이터 관리 및 Flask API 통신
class DiaryApp {
    constructor() {
        this.diaries = [];
        this.sortOrder = 'desc'; // desc: 최신순, asc: 오래된순
        this.filteredDiaries = [];
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setTodayDate();
        this.loadDiaries();
    }

    setupEventListeners() {
        // 폼 제출
        document.getElementById('diaryForm').addEventListener('submit', (e) => this.handleFormSubmit(e));

        // 검색
        document.getElementById('searchInput').addEventListener('input', (e) => this.handleSearch(e));

        // 정렬
        document.getElementById('sortBtn').addEventListener('click', () => this.toggleSort());

        // 모달 닫기
        document.querySelector('.close').addEventListener('click', () => this.closeModal());
        window.addEventListener('click', (e) => {
            const modal = document.getElementById('detailModal');
            if (e.target === modal) this.closeModal();
        });

        // 일기 내용 입력 시 자동 감정 분석
        document.getElementById('diaryContent').addEventListener('input', (e) => {
            this.debounceAnalyzeEmotion(e.target.value, 2000); // 2초 후 분석
        });
    }

    setTodayDate() {
        const today = new Date().toISOString().split('T')[0];
        document.getElementById('diaryDate').value = today;
    }

    // 디바운스 함수 (연속 입력 시 마지막 입력 후에만 실행)
    debounceAnalyzeEmotion(text, delay) {
        clearTimeout(this.analyzeTimeout);
        this.analyzeTimeout = setTimeout(() => {
            if (text.trim().length > 10) { // 10자 이상일 때만 분석
                this.analyzeEmotion(text);
            }
        }, delay);
    }

    async analyzeEmotion(text) {
        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text })
            });

            const data = await response.json();

            if (data.success) {
                this.showAIAnalysis(data);
            } else {
                console.error('감정 분석 실패:', data.error);
            }
        } catch (error) {
            console.error('감정 분석 중 오류:', error);
        }
    }

    showAIAnalysis(data) {
        const aiAnalysis = document.getElementById('aiAnalysis');
        const analyzedEmotion = document.getElementById('analyzedEmotion');
        const colorPreview = document.getElementById('colorPreview');
        const colorName = document.getElementById('colorName');
        const colorCode = document.getElementById('colorCode');
        const colorTone = document.getElementById('colorTone');

        // 분석 결과 표시
        analyzedEmotion.textContent = data.emotion;
        colorPreview.style.backgroundColor = data.color_hex;
        colorName.textContent = data.color_name;
        colorCode.textContent = data.color_hex;
        colorTone.textContent = data.tone;

        aiAnalysis.style.display = 'block';
    }

    async handleFormSubmit(e) {
        e.preventDefault();

        const date = document.getElementById('diaryDate').value;
        const title = document.getElementById('diaryTitle').value;
        const content = document.getElementById('diaryContent').value;
        const mood = document.getElementById('diaryMood').value;

        // AI 분석 결과가 있는지 확인
        const aiAnalysis = document.getElementById('aiAnalysis');
        let emotion = '';
        let colorHex = '';
        let colorName = '';

        if (aiAnalysis.style.display !== 'none') {
            emotion = document.getElementById('analyzedEmotion').textContent;
            colorHex = document.getElementById('colorCode').textContent;
            colorName = document.getElementById('colorName').textContent;
        }

        try {
            const response = await fetch('/api/save-diary', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    date: date,
                    title: title,
                    content: content,
                    mood: mood,
                    emotion: emotion,
                    color_hex: colorHex,
                    color_name: colorName
                })
            });

            const data = await response.json();

            if (data.success) {
                this.showMessage('일기가 저장되었습니다! ✨');
                
                // 폼 초기화
                document.getElementById('diaryForm').reset();
                this.setTodayDate();
                aiAnalysis.style.display = 'none';

                // 일기 목록 새로고침
                this.loadDiaries();
            } else {
                this.showMessage('일기 저장에 실패했습니다: ' + data.error, 'error');
            }
        } catch (error) {
            console.error('일기 저장 중 오류:', error);
            this.showMessage('일기 저장 중 오류가 발생했습니다.', 'error');
        }
    }

    async loadDiaries() {
        try {
            const response = await fetch('/api/diaries');
            const data = await response.json();

            if (data.success) {
                this.diaries = data.diaries;
                this.filteredDiaries = [...this.diaries];
                this.renderDiaryList();
            } else {
                console.error('일기 목록 로드 실패:', data.error);
            }
        } catch (error) {
            console.error('일기 목록 로드 중 오류:', error);
        }
    }

    handleSearch(e) {
        const searchTerm = e.target.value.toLowerCase();
        this.filteredDiaries = this.diaries.filter(diary => 
            diary.title.toLowerCase().includes(searchTerm) ||
            diary.content.toLowerCase().includes(searchTerm)
        );
        this.renderDiaryList();
    }

    toggleSort() {
        this.sortOrder = this.sortOrder === 'desc' ? 'asc' : 'desc';
        this.sortDiaries();
        this.updateSortButton();
        this.renderDiaryList();
    }

    sortDiaries() {
        this.filteredDiaries.sort((a, b) => {
            const dateA = new Date(a.created_at);
            const dateB = new Date(b.created_at);
            return this.sortOrder === 'desc' ? dateB - dateA : dateA - dateB;
        });
    }

    updateSortButton() {
        const btn = document.getElementById('sortBtn');
        btn.textContent = this.sortOrder === 'desc' ? '최신순' : '오래된순';
    }

    renderDiaryList() {
        const listContainer = document.getElementById('diaryList');

        if (this.filteredDiaries.length === 0) {
            listContainer.innerHTML = '<p class="empty-message">아직 작성한 일기가 없습니다.</p>';
            return;
        }

        const sortedDiaries = this.sortOrder === 'desc'
            ? [...this.filteredDiaries].reverse()
            : [...this.filteredDiaries];

        listContainer.innerHTML = sortedDiaries.map(diary => `
            <div class="diary-item" data-id="${diary.id}" style="border-left-color: ${diary.color_hex || '#667eea'}">
                <div class="diary-item-header">
                    <div>
                        <div class="diary-item-title">${this.escapeHtml(diary.title)}</div>
                        <div class="diary-item-date">${this.formatDate(diary.date)}</div>
                    </div>
                    <div class="diary-item-mood">${diary.mood || '기분 미선택'}</div>
                </div>
                <div class="diary-item-preview">${this.escapeHtml(diary.content)}</div>
                ${diary.emotion ? `<div class="diary-item-emotion">AI 감정: ${diary.emotion}</div>` : ''}
                <div class="diary-item-actions">
                    <button class="btn-edit" onclick="app.editDiary(${diary.id})">수정</button>
                    <button class="btn-delete" onclick="app.deleteDiary(${diary.id})">삭제</button>
                </div>
            </div>
        `).join('');

        // 일기 항목 클릭 시 전체화면 페이지로 이동
        document.querySelectorAll('.diary-item').forEach(item => {
            item.addEventListener('click', (e) => {
                if (!e.target.closest('button')) {
                    const id = parseInt(item.dataset.id);
                    window.location.href = `/diary/${id}`;
                }
            });
        });
    }

    showDiaryDetail(id) {
        const diary = this.diaries.find(d => d.id === id);
        if (!diary) return;

        const detailContent = document.getElementById('detailContent');
        const modalContent = document.querySelector('.modal-content');
        
        // 테마색 계산 (더 어두운 색상 생성)
        const themeColor = diary.color_hex || '#667eea';
        const themeColorDark = this.darkenColor(themeColor, 20);
        
        // 모달에 테마색 적용
        modalContent.style.setProperty('--theme-color', themeColor);
        modalContent.style.setProperty('--theme-color-dark', themeColorDark);
        
        // 테마색 모드 클래스 추가/제거
        if (diary.color_hex) {
            modalContent.classList.add('theme-colored');
        } else {
            modalContent.classList.remove('theme-colored');
        }
        
        detailContent.innerHTML = `
            <div class="detail-header">
                <h3>${this.escapeHtml(diary.title)}</h3>
                <div class="detail-meta">
                    <span>📅 ${this.formatDate(diary.date)}</span>
                    <span>${diary.mood || '기분 미선택'}</span>
                    ${diary.emotion ? `<span>🤖 AI: ${diary.emotion}</span>` : ''}
                    ${diary.color_hex ? `<span style="display: inline-block; width: 20px; height: 20px; background: ${diary.color_hex}; border-radius: 50%; margin-left: 10px; border: 2px solid white;"></span>` : ''}
                </div>
            </div>
            <div class="detail-body">${this.escapeHtml(diary.content)}</div>
            <div class="detail-actions">
                <button class="detail-edit" onclick="app.editDiaryFromDetail(${id})">수정</button>
                <button class="detail-delete" onclick="app.deleteDiaryFromDetail(${id})">삭제</button>
            </div>
        `;

        document.getElementById('detailModal').style.display = 'block';
    }

    editDiary(id) {
        const diary = this.diaries.find(d => d.id === id);
        if (!diary) return;

        document.getElementById('diaryDate').value = diary.date;
        document.getElementById('diaryTitle').value = diary.title;
        document.getElementById('diaryContent').value = diary.content;
        document.getElementById('diaryMood').value = diary.mood || '';

        // 기존 일기 삭제
        this.deleteDiary(id, false); // 메시지 없이 삭제

        this.showMessage('일기가 수정 모드로 열렸습니다.');
        window.scrollTo(0, 0);
    }

    editDiaryFromDetail(id) {
        this.closeModal();
        this.editDiary(id);
    }

    async deleteDiary(id, showConfirm = true) {
        if (showConfirm && !confirm('정말로 이 일기를 삭제하시겠습니까?')) {
            return;
        }

        try {
            const response = await fetch(`/api/diary/${id}`, {
                method: 'DELETE'
            });

            const data = await response.json();

            if (data.success) {
                if (showConfirm) {
                    this.showMessage('일기가 삭제되었습니다.');
                }
                this.loadDiaries();
            } else {
                this.showMessage('일기 삭제에 실패했습니다: ' + data.error, 'error');
            }
        } catch (error) {
            console.error('일기 삭제 중 오류:', error);
            this.showMessage('일기 삭제 중 오류가 발생했습니다.', 'error');
        }
    }

    deleteDiaryFromDetail(id) {
        this.closeModal();
        this.deleteDiary(id);
    }

    closeModal() {
        document.getElementById('detailModal').style.display = 'none';
    }

    showMessage(message, type = 'success') {
        const messageDiv = document.createElement('div');
        messageDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${type === 'error' ? '#ff6b6b' : '#667eea'};
            color: white;
            padding: 15px 25px;
            border-radius: 8px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
            z-index: 2000;
            animation: slideDown 0.3s ease;
            font-weight: 600;
        `;
        messageDiv.textContent = message;
        document.body.appendChild(messageDiv);

        setTimeout(() => {
            messageDiv.style.animation = 'slideUp 0.3s ease';
            setTimeout(() => messageDiv.remove(), 300);
        }, 2500);
    }

    formatDate(dateString) {
        const date = new Date(dateString + 'T00:00:00');
        const options = { year: 'numeric', month: 'long', day: 'numeric', weekday: 'short' };
        return date.toLocaleDateString('ko-KR', options);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
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
}

// 앱 초기화
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new DiaryApp();
});