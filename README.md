## Project title:

### In to the unknown

#### - Demystifying Black Box Prediction Model In Customer Default Prediction With Explainable Artificial Intelligence (XAI)

## Summary:

Advanced Machine Learning techniques (incl. Artificial Intelligence, AI) transformed the financial industry environment, laying out various opportunities as well as challenges.
Some of today's ML models offer highly accurate results at the cost of model complexity. This is sometimes referred to as Black-box models.
In today's industry, researchers and regulators are demanding techniques for making models to be human-interpretable with different kinds of reasons:

- Makes model debugging easier
- Provide transparency of the model to stakeholders
- Safeguard against bias
- Adhering to regulatory standards

Hence, this project scope follows:

1. Application of two widely used approaches for agnostic XAI:

   - The game-theoretic concept of Shapley Values, SHAP (SHapley Additive exPlanations) <sup id="ft1">[1](#f1)</sup>
   - Local surrogate models, Local Interpretable Model-Agnostic Explanation (LIME)<sup id="ft2">[2](#f2)</sup>

2. TabNet<sup id="ft4">[2](#f4)</sup> , a recently developed neural network designed specifically for tabular, non-sequential data.

The purpose of this project is to build the ML model (incl. Deep Learning) to predict customers' repayment abilities, the risk level of a given loan, alternatively known as default risk of the customer.

### Methodology:

- TabNet
  - A seuqeuntial attention mechanism for instancec-wise feature selection
  - Enables local interpretability and global interpretability hence XAI not required
- XGBoost
  - Applied XAI methodology:
    - SHAP - Global explanation
    - LIME - Instance-wise explanation

### Dataset (Considering two candidate dataset):

The main reason for initiate my project with two different dataset is that the first dataset "Home Credit" offers a large various amount of features and the total records are about (300k>) whereas the second dataset "Lending Club" offers a lower amount of features but larger records (2.5M>).

Furthermore, there are some of published literatures using Lending Club data whereas there is no specific usecase for Home Credit data in the academia.

However, it is too early to conclude which dataset to use. Hence after careful consideration in feature selection and data cleansing, I would like to decide which dataset to use.

- (1st) Home Credit:

  - Lending data for the unbanked population provided by the international consumer finance provider (<https://www.kaggle.com/c/home-credit-default-risk/overview>)
  - It provides point of sales loans, cash loans, and revolving loans to underserved borrowers
  - Various features (Before preprocessing: 150>)
  - Total records > 300k

- (2nd) Lending Club Data:
  - Peer to Peer Lending data for loans issued including the current loan status (Current, Late, Fully Paid, etc.) and latest payment information (<https://www.kaggle.com/ethon0426/lending-club-20072020q1>)
  - Various features (Before preprocessing: 140>)
  - Total records > 2.5M

## References:

- <a id="f1">1</a>: Lundberg, Scott M., and Su-In Lee. "A unified approach to interpreting model predictions." Advances in Neural Information Processing Systems. (2017).
- <a id="f2">2</a>: Ribeiro, Marco Tulio, Sameer Singh, and Carlos Guestrin. "Why should I trust you?: Explaining the predictions of any classifier." Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining. ACM (2016).
- <a id='f3'>3</a>: Mukund Sundararajan, Ankur Taly, and Qiqi Yan. 2017. Axiomatic Attribution for Deep Networks.
- <a id='f4'>3</a>: Sercan O. Arik and Tomas Pfister. 2019. TabNet: Attentive Interpretable Tabular
  Learning. (2019).

