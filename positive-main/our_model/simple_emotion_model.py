# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import os
import pickle
import sys

class SimpleEmotionAnalyzer:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.accuracy = None
        self.emotion_colors = {
            'joy': {'color': '#FFD700', 'color_name': 'golden', 'tone': 'bright'},
            'sadness': {'color': '#4682B4', 'color_name': 'blue', 'tone': 'dark'},
            'anger': {'color': '#DC143C', 'color_name': 'crimson', 'tone': 'dark'},
            'fear': {'color': '#808080', 'color_name': 'gray', 'tone': 'dark'},
            'disgust': {'color': '#9ACD32', 'color_name': 'yellow-green', 'tone': 'dark'},
            'surprise': {'color': '#FF69B4', 'color_name': 'pink', 'tone': 'bright'}
        }
        self._load_or_train_model()
    
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        
        # 한국어와 영어 모두 처리
        text = text.lower()
        
        # 한국어 전처리
        # 한국어 조사, 어미 제거 (간단한 버전)
        korean_patterns = [
            (r'이다', ''),
            (r'이다', ''),
            (r'했다', ''),
            (r'했다', ''),
            (r'했다', ''),
            (r'이다', ''),
            (r'이다', ''),
            (r'이다', ''),
        ]
        
        # 영어 전처리
        text = re.sub(r'[^a-zA-Z\s가-힣]', '', text)  # 한국어, 영어, 공백만 남김
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 한국어 키워드 매칭을 위한 추가 처리
        korean_emotion_keywords = {
            '행복': ['행복', '좋다', '좋아', '좋은', '기쁘다', '기쁜', '웃다', '웃음', '즐겁다', '즐거운', '사랑', '사랑한다', '완벽', '최고'],
            '슬픔': ['슬픔', '슬프다', '울다', '울음', '외롭다', '외로운', '우울', '상처', '아프다', '아픈', '눈물'],
            '공포': ['무섭다', '무서운', '두려움', '두려다', '겁', '겁나', '恐慌', '소름', '소름끼치다'],
            '분노': ['화', '화나다', '화나다', '짜증', '짜증나다', '화', '성나다', '미치다', '미운'],
            '역겹다': ['역겹다', '역겹다', '구역', '구역하다', '역겹다', '구역'],
            '놀라다': ['놀라다', '놀라운', '충격', '깜짝', '우와', '대박']
        }
        
        # 한국어 키워드 확인
        for emotion, keywords in korean_emotion_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return emotion
        
        # 영어 키워드 확인
        english_emotion_keywords = {
            'happiness': ['happy', 'joy', 'glad', 'excited', 'wonderful', 'amazing', 
                           'great', 'good', 'love', 'smile', 'laugh', 'fun', 'best', 'perfect'],
            'sadness': ['sad', 'cry', 'tears', 'lonely', 'depressed', 'down', 'blue',
                       'hurt', 'pain', 'sorrow', 'grief', 'miserable'],
            'fear': ['scared', 'afraid', 'worried', 'anxious', 'nervous', 'terrified',
                    'panic', 'fear', 'dread', 'horror', 'scary', 'frightened'],
            'anger': ['angry', 'mad', 'furious', 'rage', 'hate', 'annoyed', 'irritated',
                     'frustrated', 'outraged'],
            'disgust': ['disgusted', 'gross', 'sick', 'nauseated', 'revolted', 'repulsed',
                       'awful', 'terrible', 'horrible'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'wow',
                        'incredible', 'unexpected', 'startled']
        }
        
        # 영어 키워드 기반 감정 분석
        emotion_scores = {}
        for emotion, keywords in english_emotion_keywords.items():
            score = 0
            for keyword in keywords:
                score += len(re.findall(r'\b' + keyword + r'\b', text))
            emotion_scores[emotion] = score
        
        # 가장 높은 점수의 감정 반환
        if emotion_scores and max(emotion_scores.values()) > 0:
            best_emotion = max(emotion_scores, key=emotion_scores.get)
            
            # 감정 매핑
            emotion_map = {
                'happiness': 'Happiness',
                'sadness': 'Sadness', 
                'fear': 'Fear',
                'anger': 'Anger',
                'disgust': 'Disgust',
                'surprise': 'Surprise'
            }
            return emotion_map.get(best_emotion, 'Happiness')
        
        # ML 모델이 있는 경우 사용
        if self.model is not None and self.vectorizer is not None:
            try:
                # 영어 텍스트만 ML 모델에 사용
                english_text = re.sub(r'[^a-zA-Z\s]', '', text)
                if english_text.strip():
                    text_vector = self.vectorizer.transform([english_text])
                    prediction = self.model.predict(text_vector)[0]
                    
                    emotion_map = {
                        'joy': 'Happiness',
                        'sadness': 'Sadness',
                        'anger': 'Anger', 
                        'fear': 'Fear',
                        'disgust': 'Disgust',
                        'surprise': 'Surprise'
                    }
                    return emotion_map.get(prediction, 'Happiness')
            except Exception as e:
                print(f"ML 모델 분석 실패: {e}", file=sys.stderr)
        
        # 기본 키워드 분석 (영어)
        return self._keyword_analysis(text)
    
    def _load_or_train_model(self):
        csv_path = os.path.join(os.path.dirname(__file__), 'emotion_sentimen_dataset.csv')
        model_cache_path = os.path.join(os.path.dirname(__file__), 'model_cache.pkl')
        
        if os.path.exists(model_cache_path):
            try:
                with open(model_cache_path, 'rb') as f:
                    cached = pickle.load(f)
                    self.vectorizer = cached['vectorizer']
                    self.model = cached['model']
                    self.accuracy = cached['accuracy']
                print("OK: Loaded cached model!")
                return
            except Exception as e:
                print(f"WARN: Cache load failed: {e}")
        
        if os.path.exists(csv_path):
            try:
                self._train_model_from_dataset(csv_path)
                self._save_model_cache(model_cache_path)
                print("OK: ML model trained and cached!")
                return
            except Exception as e:
                print(f"WARN: Dataset training failed: {e}")
        
        self._setup_default_model()
    
    def _train_model_from_dataset(self, csv_path):
        print("Loading dataset...", file=sys.stderr)
        df = pd.read_csv(csv_path, encoding='latin1', on_bad_lines='skip')
        
        try:
            df_renamed = df.rename(columns={'Emotion': 'label', 'text': 'text'})
            df_clean = df_renamed[['text', 'label']].copy()
        except KeyError:
            raise ValueError("Dataset must have 'Emotion' and 'text' columns")
        
        print("Cleaning text...", file=sys.stderr)
        df_clean['text'] = df_clean['text'].apply(self.clean_text)
        df_clean = df_clean.dropna(subset=['text', 'label'])
        df_clean = df_clean[df_clean['text'] != ""]
        
        label_map = {
            'happiness': 'joy', 'fun': 'joy', 'enthusiasm': 'joy',
            'relief': 'joy', 'love': 'joy',
            'sadness': 'sadness', 'empty': 'sadness', 'boredom': 'sadness',
            'anger': 'anger', 'worry': 'fear', 'hate': 'disgust',
            'surprise': 'surprise'
        }
        
        df_clean['label'] = df_clean['label'].map(label_map)
        df_clean = df_clean.dropna(subset=['label'])
        
        print(f"Training with {len(df_clean)} samples", file=sys.stderr)
        
        X = df_clean['text']
        y = df_clean['label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        self.model = LogisticRegression(
            max_iter=1000, random_state=42, class_weight='balanced'
        )
        self.model.fit(X_train_tfidf, y_train)
        
        y_pred = self.model.predict(X_test_tfidf)
        from sklearn.metrics import accuracy_score
        self.accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {self.accuracy * 100:.2f}%", file=sys.stderr)
    
    def _save_model_cache(self, cache_path):
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'vectorizer': self.vectorizer,
                    'model': self.model,
                    'accuracy': self.accuracy
                }, f)
        except Exception as e:
            print(f"WARN: Could not cache model: {e}", file=sys.stderr)
    
    def _setup_default_model(self):
        self.model = None
        self.vectorizer = None
        print("WARN: Using keyword fallback", file=sys.stderr)
    
    def analyze_emotion(self, text):
        """메인 감정 분석 함수"""
        if not isinstance(text, str) or not text.strip():
            return 'Happiness'
        
        # 1. 한국어 키워드 우선 분석
        korean_result = self._analyze_korean_emotion(text)
        if korean_result:
            return korean_result
        
        # 2. 영어 키워드 분석
        english_result = self._analyze_english_emotion(text)
        if english_result:
            return english_result
        
        # 3. ML 모델 사용 (영어 텍스트만)
        ml_result = self._analyze_with_ml(text)
        if ml_result:
            return ml_result
        
        # 4. 기본값
        return 'Happiness'
    
    def _analyze_korean_emotion(self, text):
        """한국어 텍스트 감정 분석"""
        text_lower = text.lower()
        
        # 한국어 감정 키워드 (더 포괄적으로)
        korean_emotions = {
            'Fear': [
                '무서워', '무섭다', '무서운', '두려워', '두려운', '겁', '겁나', '겁나는',
                '恐慌', '소름', '소름끼치다', '무서움', '무서웠다', '무서웠다',
                '무서워서', '무서워서', '무서워서', '무서워서'
            ],
            'Happiness': [
                '행복', '행복해', '행복한', '좋아', '좋다', '좋은', '기쁘다', '기쁜',
                '웃다', '웃음', '즐겁다', '즐거운', '사랑', '사랑해', '완벽', '최고',
                '행복했다', '행복했다', '행복했다', '행복했다'
            ],
            'Sadness': [
                '슬프다', '슬픈', '울다', '울음', '외롭다', '외로운', '우울', '상처',
                '아프다', '아픈', '눈물', '슬픔', '울었다', '울었다'
            ],
            'Anger': [
                '화', '화나', '화나다', '짜증', '짜증나', '짜증나다', '화', '성나',
                '미치다', '미운', '화났다', '화났다'
            ],
            'Disgust': [
                '역겹다', '역겹다', '구역', '구역하다', '역겹다', '구역'
            ],
            'Surprise': [
                '놀라다', '놀라운', '충격', '깜짝', '우와', '대박', '놀랐다', '놀랐다'
            ]
        }
        
        # 키워드 매칭
        for emotion, keywords in korean_emotions.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return emotion
        
        return None
    
    def _analyze_english_emotion(self, text):
        """영어 텍스트 감정 분석"""
        text_lower = text.lower()
        
        # 영어 감정 키워드 (더 포괄적으로)
        english_emotions = {
            'Fear': [
                'scared', 'afraid', 'worried', 'anxious', 'nervous', 'terrified',
                'panic', 'fear', 'dread', 'horror', 'scary', 'frightened', 'frightening',
                'strange', 'noise', 'pounding', 'cant sleep', 'cant sleep', 'sleep',
                'shivering', 'trembling', 'uneasy', 'uncomfortable', 'threat'
            ],
            'Happiness': [
                'happy', 'joy', 'glad', 'excited', 'wonderful', 'amazing', 
                'great', 'good', 'love', 'smile', 'laugh', 'fun', 'best', 'perfect',
                'sun', 'shining', 'aced', 'test', 'favorite', 'song', 'wonderful day'
            ],
            'Sadness': [
                'sad', 'cry', 'tears', 'lonely', 'depressed', 'down', 'blue',
                'hurt', 'pain', 'sorrow', 'grief', 'miserable', 'lonely'
            ],
            'Anger': [
                'angry', 'mad', 'furious', 'rage', 'hate', 'annoyed', 'irritated',
                'frustrated', 'outraged', 'pissed', 'livid'
            ],
            'Disgust': [
                'disgusted', 'gross', 'sick', 'nauseated', 'revolted', 'repulsed',
                'awful', 'terrible', 'horrible', 'disgusting'
            ],
            'Surprise': [
                'surprised', 'shocked', 'amazed', 'astonished', 'wow',
                'incredible', 'unexpected', 'startled', 'suddenly'
            ]
        }
        
        # 키워드 매칭 점수 계산
        emotion_scores = {}
        for emotion, keywords in english_emotions.items():
            score = 0
            for keyword in keywords:
                # 전체 단어 매칭
                if keyword in text_lower:
                    score += 1
                # 부분 단어 매칭 (예: "scary movie"에서 "scary" 매칭)
                words = text_lower.split()
                for word in words:
                    if keyword in word or word in keyword:
                        score += 0.5
            emotion_scores[emotion] = score
        
        # 가장 높은 점수의 감정 반환
        if emotion_scores and max(emotion_scores.values()) > 0:
            return max(emotion_scores, key=emotion_scores.get)
        
        return None
    
    def _analyze_with_ml(self, text):
        """ML 모델을 사용한 감정 분석"""
        if self.model is not None and self.vectorizer is not None:
            try:
                # 영어 텍스트만 ML 모델에 사용
                english_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
                if english_text.strip() and len(english_text.split()) >= 3:  # 최소 3개 단어
                    text_vector = self.vectorizer.transform([english_text])
                    prediction = self.model.predict(text_vector)[0]
                    
                    emotion_map = {
                        'joy': 'Happiness',
                        'sadness': 'Sadness',
                        'anger': 'Anger', 
                        'fear': 'Fear',
                        'disgust': 'Disgust',
                        'surprise': 'Surprise'
                    }
                    return emotion_map.get(prediction, 'Happiness')
            except Exception as e:
                print(f"ML 모델 분석 실패: {e}", file=sys.stderr)
        
        return None
    
    def _load_or_train_model(self):
        csv_path = os.path.join(os.path.dirname(__file__), 'emotion_sentimen_dataset.csv')
        model_cache_path = os.path.join(os.path.dirname(__file__), 'model_cache.pkl')
        
        if os.path.exists(model_cache_path):
            try:
                with open(model_cache_path, 'rb') as f:
                    cached = pickle.load(f)
                    self.vectorizer = cached['vectorizer']
                    self.model = cached['model']
                    self.accuracy = cached['accuracy']
                print("OK: Loaded cached model!")
                return
            except Exception as e:
                print(f"WARN: Cache load failed: {e}")
        
        if os.path.exists(csv_path):
            try:
                self._train_model_from_dataset(csv_path)
                self._save_model_cache(model_cache_path)
                print("OK: ML model trained and cached!")
                return
            except Exception as e:
                print(f"WARN: Dataset training failed: {e}")
        
        self._setup_default_model()
    
    def _train_model_from_dataset(self, csv_path):
        print("Loading dataset...", file=sys.stderr)
        df = pd.read_csv(csv_path, encoding='latin1', on_bad_lines='skip')
        
        try:
            df_renamed = df.rename(columns={'Emotion': 'label', 'text': 'text'})
            df_clean = df_renamed[['text', 'label']].copy()
        except KeyError:
            raise ValueError("Dataset must have 'Emotion' and 'text' columns")
        
        print("Cleaning text...", file=sys.stderr)
        df_clean['text'] = df_clean['text'].apply(self.clean_text)
        df_clean = df_clean.dropna(subset=['text', 'label'])
        df_clean = df_clean[df_clean['text'] != ""]
        
        label_map = {
            'happiness': 'joy', 'fun': 'joy', 'enthusiasm': 'joy',
            'relief': 'joy', 'love': 'joy',
            'sadness': 'sadness', 'empty': 'sadness', 'boredom': 'sadness',
            'anger': 'anger', 'worry': 'fear', 'hate': 'disgust',
            'surprise': 'surprise'
        }
        
        df_clean['label'] = df_clean['label'].map(label_map)
        df_clean = df_clean.dropna(subset=['label'])
        
        print(f"Training with {len(df_clean)} samples", file=sys.stderr)
        
        X = df_clean['text']
        y = df_clean['label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        self.model = LogisticRegression(
            max_iter=1000, random_state=42, class_weight='balanced'
        )
        self.model.fit(X_train_tfidf, y_train)
        
        y_pred = self.model.predict(X_test_tfidf)
        from sklearn.metrics import accuracy_score
        self.accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {self.accuracy * 100:.2f}%", file=sys.stderr)
    
    def _save_model_cache(self, cache_path):
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'vectorizer': self.vectorizer,
                    'model': self.model,
                    'accuracy': self.accuracy
                }, f)
        except Exception as e:
            print(f"WARN: Could not cache model: {e}", file=sys.stderr)
    
    def _setup_default_model(self):
        self.model = None
        self.vectorizer = None
        print("WARN: Using keyword fallback", file=sys.stderr)
    
    def analyze_emotion(self, text):
        """메인 감정 분석 함수"""
        if not isinstance(text, str) or not text.strip():
            return 'Happiness'
        
        # 1. 한국어 키워드 우선 분석
        korean_result = self._analyze_korean_emotion(text)
        if korean_result:
            return korean_result
        
        # 2. 영어 키워드 분석
        english_result = self._analyze_english_emotion(text)
        if english_result:
            return english_result
        
        # 3. ML 모델 사용 (영어 텍스트만)
        ml_result = self._analyze_with_ml(text)
        if ml_result:
            return ml_result
        
        # 4. 기본값
        return 'Happiness'
    
    def _analyze_korean_emotion(self, text):
        """한국어 텍스트 감정 분석"""
        text_lower = text.lower()
        
        # 한국어 감정 키워드 (더 포괄적으로)
        korean_emotions = {
            'Fear': [
                '무서워', '무섭다', '무서운', '두려워', '두려운', '겁', '겁나', '겁나는',
                '恐慌', '소름', '소름끼치다', '무서움', '무서웠다', '무서웠다',
                '무서워서', '무서워서', '무서워서', '무서워서'
            ],
            'Happiness': [
                '행복', '행복해', '행복한', '좋아', '좋다', '좋은', '기쁘다', '기쁜',
                '웃다', '웃음', '즐겁다', '즐거운', '사랑', '사랑해', '완벽', '최고',
                '행복했다', '행복했다', '행복했다', '행복했다'
            ],
            'Sadness': [
                '슬프다', '슬픈', '울다', '울음', '외롭다', '외로운', '우울', '상처',
                '아프다', '아픈', '눈물', '슬픔', '울었다', '울었다'
            ],
            'Anger': [
                '화', '화나', '화나다', '짜증', '짜증나', '짜증나다', '화', '성나',
                '미치다', '미운', '화났다', '화났다'
            ],
            'Disgust': [
                '역겹다', '역겹다', '구역', '구역하다', '역겹다', '구역'
            ],
            'Surprise': [
                '놀라다', '놀라운', '충격', '깜짝', '우와', '대박', '놀랐다', '놀랐다'
            ]
        }
        
        # 키워드 매칭
        for emotion, keywords in korean_emotions.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return emotion
        
        return None
    
    def _analyze_english_emotion(self, text):
        """영어 텍스트 감정 분석"""
        text_lower = text.lower()
        
        # 영어 감정 키워드 (더 포괄적으로)
        english_emotions = {
            'Fear': [
                'scared', 'afraid', 'worried', 'anxious', 'nervous', 'terrified',
                'panic', 'fear', 'dread', 'horror', 'scary', 'frightened', 'frightening',
                'strange', 'noise', 'pounding', 'cant sleep', 'cant sleep', 'sleep',
                'shivering', 'trembling', 'uneasy', 'uncomfortable', 'threat'
            ],
            'Happiness': [
                'happy', 'joy', 'glad', 'excited', 'wonderful', 'amazing', 
                'great', 'good', 'love', 'smile', 'laugh', 'fun', 'best', 'perfect',
                'sun', 'shining', 'aced', 'test', 'favorite', 'song', 'wonderful day'
            ],
            'Sadness': [
                'sad', 'cry', 'tears', 'lonely', 'depressed', 'down', 'blue',
                'hurt', 'pain', 'sorrow', 'grief', 'miserable', 'lonely'
            ],
            'Anger': [
                'angry', 'mad', 'furious', 'rage', 'hate', 'annoyed', 'irritated',
                'frustrated', 'outraged', 'pissed', 'livid'
            ],
            'Disgust': [
                'disgusted', 'gross', 'sick', 'nauseated', 'revolted', 'repulsed',
                'awful', 'terrible', 'horrible', 'disgusting'
            ],
            'Surprise': [
                'surprised', 'shocked', 'amazed', 'astonished', 'wow',
                'incredible', 'unexpected', 'startled', 'suddenly'
            ]
        }
        
        # 키워드 매칭 점수 계산
        emotion_scores = {}
        for emotion, keywords in english_emotions.items():
            score = 0
            for keyword in keywords:
                # 전체 단어 매칭
                if keyword in text_lower:
                    score += 1
                # 부분 단어 매칭 (예: "scary movie"에서 "scary" 매칭)
                words = text_lower.split()
                for word in words:
                    if keyword in word or word in keyword:
                        score += 0.5
            emotion_scores[emotion] = score
        
        # 가장 높은 점수의 감정 반환
        if emotion_scores and max(emotion_scores.values()) > 0:
            return max(emotion_scores, key=emotion_scores.get)
        
        return None
    
    def _analyze_with_ml(self, text):
        """ML 모델을 사용한 감정 분석"""
        if self.model is not None and self.vectorizer is not None:
            try:
                # 영어 텍스트만 ML 모델에 사용
                english_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
                if english_text.strip() and len(english_text.split()) >= 3:  # 최소 3개 단어
                    text_vector = self.vectorizer.transform([english_text])
                    prediction = self.model.predict(text_vector)[0]
                    
                    emotion_map = {
                        'joy': 'Happiness',
                        'sadness': 'Sadness',
                        'anger': 'Anger', 
                        'fear': 'Fear',
                        'disgust': 'Disgust',
                        'surprise': 'Surprise'
                    }
                    return emotion_map.get(prediction, 'Happiness')
            except Exception as e:
                print(f"ML 모델 분석 실패: {e}", file=sys.stderr)
        
        return None
    
    def analyze_emotion_and_color(self, diary_entry, show_visualization=False):
        emotion = self.analyze_emotion(diary_entry)
        result = self.get_color_recommendation(emotion)
        print(f"Emotion: {emotion}")
        return result

analyzer = SimpleEmotionAnalyzer()

def analyze_emotion_and_color(diary_entry, show_visualization=False):
    return analyzer.analyze_emotion_and_color(diary_entry, show_visualization)

if __name__ == "__main__":
    test_texts = [
        ("I am so happy today!", "Happiness"),
        ("I feel sad and lonely", "Sadness"),
        ("I'm so angry", "Anger"),
    ]
    
    for text, expected in test_texts:
        result = analyzer.analyze_emotion_and_color(text)
        actual = result.get('emotion')
        print(f"[{'OK' if actual == expected else 'FAIL'}] {text} -> {actual}")
