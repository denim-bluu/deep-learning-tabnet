# ST449-project-2021

This is the repository for the course project. Please keep anything project related in this repository.

**Important dates**:

- **Project proposal deadline**: 14th March, 5pm London UK time (fill in the information below by this date)
- **First notification deadline**: 28th March, 5pm London UK time (approval of project topic or request for revision by this date)
- **Project topic approval deadline**: 31st March, 5pm London UK time (all project topics must be approved by this date)
- **Project solution submission deadline**: 30th April, 5pm London UK time

Your first task is the propose your project topic by filling the information requested below.

You are encouraged to propose your course project topic as soon as possible but not later than by the project proposal deadline indicated above. We will try to process your project proposal and provide feedback as soon as possible. In any case, the first feedback from us will not be later than by the first notification deadline indicated above. All project topics must be approved by the project approval deadline indicated above -- the approval will be indicated in the feedback section below.

You will receive feedback for your project topic proposal in the feedback section below of this Markdown file. A project topic proposal may be immediately approved or some revision may be required. The feedback should be limited to a few rounds of interactions (one or two) in order to deal with the workload.

---

Please add the following information:

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

## Feedback:

- [MV 6 April 2021] Revised proposal approved. I trust your project will have a focus on deep learning / neural network models. You may want to make sure to well explain the methods that you study, especially in relation of _neural network architectures_ and _interpretability_. 
- [MV, 27 March 2021] Approved. Sounds interesting. You have identified a good set of references. You may want to focus on explaining the underlying methodology used, in particular, for explaining the predictions of a classifier.

---

## Candidate project topics:

Here you may find information about some candidate project topics: [Project.md](https://github.com/lse-st449/lectures2021/blob/master/Projects.md).

**Important**: You do not need to take a project topic listed in our list of suggestions -- you are encourged to come up with a project topic proposal of your own, which is not listed in our list.

## Marking criteria:

<img src="https://github.com/lse-st449/lectures2021/blob/main/images/ST449-final-coursework-rubric.png"></img>
