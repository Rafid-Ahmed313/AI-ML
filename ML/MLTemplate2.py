import warnings
warnings.filterwarnings("ignore")

from lazypredict.Supervised import LazyClassifier, LazyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_breast_cancer, fetch_california_housing
import torch
import pandas as pd

# ============================
# Classification
# ============================
X_cls, y_cls = load_breast_cancer(return_X_y=True)
X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

lazy_cls = LazyClassifier(verbose=0, ignore_warnings=True)
models_cls, _ = lazy_cls.fit(X_cls_train, X_cls_test, y_cls_train, y_cls_test)
top4_cls = models_cls.head(4).index.tolist()

model_map_cls = {
    "LogisticRegression": LogisticRegression(),
    "SVC": SVC(probability=True),
    "RandomForestClassifier": RandomForestClassifier(),
    "GradientBoostingClassifier": GradientBoostingClassifier(),
    "GaussianNB": GaussianNB(),
    "KNeighborsClassifier": KNeighborsClassifier(),
}

voting_cls = VotingClassifier(estimators=[(name, model_map_cls[name]) for name in top4_cls], voting='soft')
voting_cls.fit(X_cls_train, y_cls_train)
acc_voting_cls = accuracy_score(y_cls_test, voting_cls.predict(X_cls_test))

xgb_stack_cls = StackingClassifier(
    estimators=[(f'xgb_{i}', XGBClassifier(eval_metric='logloss')) for i in range(4)],
    final_estimator=LogisticRegression()
)
xgb_stack_cls.fit(X_cls_train, y_cls_train)
acc_stack_cls = accuracy_score(y_cls_test, xgb_stack_cls.predict(X_cls_test))

# ============================
# Regression
# ============================
X_reg, y_reg = fetch_california_housing(return_X_y=True)
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

lazy_reg = LazyRegressor(verbose=0, ignore_warnings=True)
models_reg, _ = lazy_reg.fit(X_reg_train, X_reg_test, y_reg_train, y_reg_test)
top4_reg = models_reg.head(4).index.tolist()

model_map_reg = {
    "LinearRegression": LinearRegression(),
    "SVR": SVR(),
    "RandomForestRegressor": RandomForestRegressor(),
    "GradientBoostingRegressor": GradientBoostingRegressor(),
    "KNeighborsRegressor": KNeighborsRegressor(),
}

voting_reg = VotingRegressor(estimators=[(name, model_map_reg[name]) for name in top4_reg])
voting_reg.fit(X_reg_train, y_reg_train)
r2_voting_reg = r2_score(y_reg_test, voting_reg.predict(X_reg_test))

xgb_stack_reg = StackingRegressor(
    estimators=[(f'xgb_{i}', XGBRegressor()) for i in range(4)],
    final_estimator=LinearRegression()
)
xgb_stack_reg.fit(X_reg_train, y_reg_train)
r2_stack_reg = r2_score(y_reg_test, xgb_stack_reg.predict(X_reg_test))

# ============================
# Results
# ============================
results = pd.DataFrame({
    'Model': ["VotingClassifier", "XGBStackClassifier", "VotingRegressor", "XGBStackRegressor"],
    'Metric': ["Accuracy", "Accuracy", "R^2", "R^2"],
    'Score': [acc_voting_cls, acc_stack_cls, r2_voting_reg, r2_stack_reg]
})

print("\nFinal Results:")
print(results)

