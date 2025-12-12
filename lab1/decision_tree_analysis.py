import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Настройка для корректного отображения русского текста
import matplotlib
matplotlib.rcParams['font.family'] = ['DejaVu Sans'] 
matplotlib.rcParams['axes.unicode_minus'] = False
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

def load_and_explore_data():
    """Загрузка и исследование данных Adult Income"""
    print("=== Информация о наборе данных Adult Income ===")
    
    try:
        # Загрузка CSV данных
        df = pd.read_csv('adult_income.csv')
        
        print(f"Название датасета: Adult Income Dataset (UCI)")
        print(f"Количество образцов: {df.shape[0]}, Количество признаков: {df.shape[1] - 1}")
        print(f"Распределение целевой переменной:")
        print(df['income'].value_counts())
        
        return df
        
    except Exception as e:
        print(f"Ошибка загрузки CSV данных: {e}")
        print("Создаем синтетический датасет...")
        
        # Альтернативный вариант - создаем синтетический датасет
        np.random.seed(42)
        n_samples = 10000
        n_features = 10
        
        # Создаем синтетические данные
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.5 > 0).astype(int)
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        df['income'] = y
        
        print(f"Синтетический датасет: {n_samples} образцов, {n_features} признаков")
        print(f"Распределение целевой переменной: {df['income'].value_counts().to_dict()}")
        
        return df

def prepare_data(df):
    """Подготовка данных для обучения"""
    print("\n=== Подготовка данных ===")
    
    if 'workclass' in df.columns:
        # Выбираем числовые признаки
        numeric_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 
                          'capital-loss', 'hours-per-week']
        
        # Кодируем категориальные признаки
        categorical_features = ['workclass', 'education', 'marital-status', 
                              'occupation', 'relationship', 'race', 'sex', 'native-country']
        
        # Создаем финальный датасет
        df_processed = df[numeric_features + ['income']].copy()
        
        # Добавляем закодированные категориальные признаки
        le = LabelEncoder()
        for feature in categorical_features:
            df_processed[f'{feature}_encoded'] = le.fit_transform(df[feature].astype(str))
        
        # Разделяем на признаки и целевую переменную
        X = df_processed.drop('income', axis=1)
        y = df_processed['income']
        
    else:
        # Для синтетического датасета
        X = df.drop('income', axis=1)
        y = df['income']
    
    # Кодируем целевую переменную
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")
    print(f"Количество признаков: {X.shape[1]}")
    
    return X_train, X_test, y_train, y_test, X, y


def build_unrestricted_tree(X_train, X_test, y_train, y_test):
    """Построение дерева решений без ограничений"""
    print("\n=== Дерево решений без ограничений ===")
    
    dt_unrestricted = DecisionTreeClassifier(random_state=42)
    dt_unrestricted.fit(X_train, y_train)
    
    y_pred_test = dt_unrestricted.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    cv_scores = cross_val_score(dt_unrestricted, X_train, y_train, cv=5)
    
    print(f"Точность на тестовой выборке: {test_accuracy:.4f}")
    print(f"Кросс-валидация: {cv_scores.mean():.4f}")
    print(f"Глубина дерева: {dt_unrestricted.get_depth()}")
    print(f"Количество листьев: {dt_unrestricted.get_n_leaves()}")
    
    # Подробный отчет
    print(f"\nОтчет о классификации (неограниченное дерево):")
    print(classification_report(y_test, y_pred_test, target_names=['<=50K', '>50K']))
    
    # Матрица ошибок
    print(f"\nМатрица ошибок (неограниченное дерево):")
    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)
    
    return dt_unrestricted, test_accuracy, cv_scores.mean()

def build_restricted_tree(X_train, X_test, y_train, y_test):
    """Построение дерева решений с ограничениями"""
    print("\n=== Дерево решений с ограничениями ===")
    
    dt_restricted = DecisionTreeClassifier(
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features=None,
        min_impurity_decrease=0.0001,
        criterion='gini',
        splitter='best',
        random_state=42
    )
    
    dt_restricted.fit(X_train, y_train)
    
    y_pred_test = dt_restricted.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    cv_scores = cross_val_score(dt_restricted, X_train, y_train, cv=5)
    
    print(f"Точность на тестовой выборке: {test_accuracy:.4f}")
    print(f"Кросс-валидация: {cv_scores.mean():.4f}")
    print(f"Глубина дерева: {dt_restricted.get_depth()}")
    print(f"Количество листьев: {dt_restricted.get_n_leaves()}")
    
    # Подробный отчет
    print(f"\nОтчет о классификации (ограниченное дерево):")
    print(classification_report(y_test, y_pred_test, target_names=['<=50K', '>50K']))
    
    # Матрица ошибок
    print(f"\nМатрица ошибок (ограниченное дерево):")
    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)
    
    return dt_restricted, test_accuracy, cv_scores.mean()

def compare_models(dt_unrestricted, dt_restricted, test_acc_unrestricted, test_acc_restricted, cv_unrestricted, cv_restricted):
    """Сравнение моделей"""
    print("\n=== Сравнение моделей ===")
    
    print(f"Модель без ограничений:")
    print(f"  Тестовая точность: {test_acc_unrestricted:.4f}")
    print(f"  Кросс-валидация: {cv_unrestricted:.4f}")
    print(f"  Глубина: {dt_unrestricted.get_depth()}")
    print(f"  Листья: {dt_unrestricted.get_n_leaves()}")
    
    print(f"\nМодель с ограничениями:")
    print(f"  Тестовая точность: {test_acc_restricted:.4f}")
    print(f"  Кросс-валидация: {cv_restricted:.4f}")
    print(f"  Глубина: {dt_restricted.get_depth()}")
    print(f"  Листья: {dt_restricted.get_n_leaves()}")
    
    print(f"\n=== Результат сравнения ===")
    if test_acc_restricted > test_acc_unrestricted:
        improvement = test_acc_restricted - test_acc_unrestricted
        print(f"OK: Ограниченная модель ЛУЧШЕ на {improvement:.4f} ({improvement*100:.2f}%)")
    else:
        decline = test_acc_unrestricted - test_acc_restricted
        print(f"BAD: Ограниченная модель ХУЖЕ на {decline:.4f} ({decline*100:.2f}%)")

def analyze_feature_importance(dt_restricted, X):
    """Анализ важности признаков"""
    print("\n=== Анализ важности признаков ===")
    
    feature_importance = dt_restricted.feature_importances_
    importance_df = pd.DataFrame({
        'Признак': X.columns,
        'Важность': feature_importance
    }).sort_values('Важность', ascending=False)
    
    print("Топ-5 важных признаков:")
    for i, (_, row) in enumerate(importance_df.head(5).iterrows(), 1):
        print(f"{i}. {row['Признак']}: {row['Важность']:.4f}")
    
    return importance_df

def visualize_decision_tree(dt_restricted, X):
    """Визуализация полного дерева решений с ограничениями"""
    print("\n=== Визуализация дерева решений ===")
    
    plt.figure(figsize=(60, 40))
    plot_tree(dt_restricted, 
              feature_names=X.columns,
              class_names=['<=50K', '>50K'],
              filled=True,
              rounded=True,
              fontsize=8,
              max_depth=None,
              impurity=True,
              proportion=True)
    plt.title('Полное дерево решений с ограничениями', fontsize=20, pad=30)
    plt.tight_layout()
    plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("OK: Визуализация сохранена в файл 'decision_tree.png'")

def main():
    """Основная функция"""
    print("=== Сравнение деревьев решений на датасете Adult Income ===")
    
    # Загрузка и подготовка данных
    df = load_and_explore_data()
    X_train, X_test, y_train, y_test, X, y = prepare_data(df)
    
    # Построение неограниченного дерева
    dt_unrestricted, test_acc_unrestricted, cv_unrestricted = build_unrestricted_tree(
        X_train, X_test, y_train, y_test
    )
    
    # Построение ограниченного дерева
    dt_restricted, test_acc_restricted, cv_restricted = build_restricted_tree(
        X_train, X_test, y_train, y_test
    )
    
    # Сравнение моделей
    compare_models(dt_unrestricted, dt_restricted, test_acc_unrestricted, test_acc_restricted, cv_unrestricted, cv_restricted)
    
    # Анализ важности признаков
    importance_df = analyze_feature_importance(dt_restricted, X)
    
    # Визуализация дерева решений
    visualize_decision_tree(dt_restricted, X)

if __name__ == "__main__":
    main()
