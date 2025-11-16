// 일기 데이터 관리
class DiaryApp {
    constructor() {
        this.diaries = this.loadFromLocalStorage();
        this.sortOrder = 'desc'; // desc: 최신순, asc: 오래된순
        this.filteredDiaries = [...this.diaries];
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setTodayDate();
        this.renderDiaryList();
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
    }

    setTodayDate() {
        const today = new Date().toISOString().split('T')[0];
        document.getElementById('diaryDate').value = today;
    }

    handleFormSubmit(e) {
        e.preventDefault();

        const date = document.getElementById('diaryDate').value;
        const title = document.getElementById('diaryTitle').value;
        const content = document.getElementById('diaryContent').value;
        const mood = document.getElementById('diaryMood').value;

        // 새 일기 객체 생성
        const newDiary = {
            id: Date.now(),
            date: date,
            title: title,
            content: content,
            mood: mood,
            createdAt: new Date().toISOString()
        };

        this.diaries.unshift(newDiary);
        this.saveToLocalStorage();
        this.renderDiaryList();

        // 폼 초기화
        document.getElementById('diaryForm').reset();
        this.setTodayDate();

        // 성공 메시지
        this.showMessage('일기가 저장되었습니다! ✨');
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
            const dateA = new Date(a.date);
            const dateB = new Date(b.date);
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
            <div class="diary-item" data-id="${diary.id}">
                <div class="diary-item-header">
                    <div>
                        <div class="diary-item-title">${this.escapeHtml(diary.title)}</div>
                        <div class="diary-item-date">${this.formatDate(diary.date)}</div>
                    </div>
                    <div class="diary-item-mood">${diary.mood}</div>
                </div>
                <div class="diary-item-preview">${this.escapeHtml(diary.content)}</div>
                <div class="diary-item-actions">
                    <button class="btn-edit" onclick="app.editDiary(${diary.id})">수정</button>
                    <button class="btn-delete" onclick="app.deleteDiary(${diary.id})">삭제</button>
                </div>
            </div>
        `).join('');

        // 일기 항목 클릭 시 상세보기
        document.querySelectorAll('.diary-item').forEach(item => {
            item.addEventListener('click', (e) => {
                if (!e.target.closest('button')) {
                    const id = parseInt(item.dataset.id);
                    this.showDiaryDetail(id);
                }
            });
        });
    }

    showDiaryDetail(id) {
        const diary = this.diaries.find(d => d.id === id);
        if (!diary) return;

        const detailContent = document.getElementById('detailContent');
        detailContent.innerHTML = `
            <div class="detail-header">
                <h3>${this.escapeHtml(diary.title)}</h3>
                <div class="detail-meta">
                    <span>📅 ${this.formatDate(diary.date)}</span>
                    <span>${diary.mood} 기분</span>
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
        document.getElementById('diaryMood').value = diary.mood;

        // 기존 일기 삭제
        this.diaries = this.diaries.filter(d => d.id !== id);
        this.saveToLocalStorage();
        this.renderDiaryList();

        this.showMessage('일기가 수정 모드로 열렸습니다.');
        window.scrollTo(0, 0);
    }

    editDiaryFromDetail(id) {
        this.closeModal();
        this.editDiary(id);
    }

    deleteDiary(id) {
        if (confirm('정말로 이 일기를 삭제하시겠습니까?')) {
            this.diaries = this.diaries.filter(d => d.id !== id);
            this.saveToLocalStorage();
            this.renderDiaryList();
            this.showMessage('일기가 삭제되었습니다.');
        }
    }

    deleteDiaryFromDetail(id) {
        this.closeModal();
        this.deleteDiary(id);
    }

    closeModal() {
        document.getElementById('detailModal').style.display = 'none';
    }

    showMessage(message) {
        const messageDiv = document.createElement('div');
        messageDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #667eea;
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

    saveToLocalStorage() {
        localStorage.setItem('diaries', JSON.stringify(this.diaries));
    }

    loadFromLocalStorage() {
        const stored = localStorage.getItem('diaries');
        return stored ? JSON.parse(stored) : [];
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
}

// 앱 초기화
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new DiaryApp();
});
