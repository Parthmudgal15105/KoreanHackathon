"""Input data for the resume matching engine."""

SKILL_ALIASES = {
    "python": "python",
    "pyhton": "python",
    "java": "java",
    "javascript": "javascript",
    "javascrpit": "javascript",
    "js": "javascript",
    "typescript": "typescript",
    "typescrpit": "typescript",
    "c++": "cpp",
    "cpp": "cpp",
    "r": "r",
    "kotlin": "kotlin",

    "machinelearning": "machine_learning",
    "machine learning": "machine_learning",
    "ml": "machine_learning",
    "sklearn": "machine_learning",
    "deeplearning": "deep_learning",
    "deep learning": "deep_learning",
    "deep-learning": "deep_learning",
    "tensorflow": "tensorflow",
    "pytorch": "pytorch",
    "keras": "keras",
    "nlp": "nlp",
    "bert": "bert",
    "xgboost": "xgboost",
    "feature engineering": "feature_engineering",
    "statistics": "statistics",
    "stats": "statistics",
    "regression": "regression",
    "clustering": "clustering",
    "data-viz": "data_visualization",
    "data visualization": "data_visualization",
    "data viz": "data_visualization",
    "matplotlib": "data_visualization",
    "tableau": "data_visualization",
    "power-bi": "data_visualization",
    "power bi": "data_visualization",
    "powerbi": "data_visualization",
    "pandas": "pandas",
    "numpy": "numpy",

    "react": "react",
    "reacts": "react",
    "reactjs": "react",
    "vue": "vue",
    "vue.js": "vue",
    "vuejs": "vue",
    "redux": "redux",
    "tailwind": "tailwind",
    "html/css": "html_css",
    "html css": "html_css",
    "html": "html_css",
    "css": "html_css",
    "jest": "jest",
    "graphql": "graphql",

    "node.js": "nodejs",
    "nodejs": "nodejs",
    "node js": "nodejs",
    "flask": "flask",
    "spring boot": "spring_boot",
    "springboot": "spring_boot",
    "rest api": "rest_api",
    "rest": "rest_api",
    "restapi": "rest_api",
    "microservices": "microservices",

    "sql": "sql",
    "mysql": "mysql",
    "mysq": "mysql",
    "postgresql": "postgresql",
    "postgres": "postgresql",
    "mongodb": "mongodb",
    "redis": "redis",

    "docker": "docker",
    "kubernetes": "kubernetes",
    "kubernates": "kubernetes",
    "k8s": "kubernetes",
    "ci/cd": "ci_cd",
    "cicd": "ci_cd",
    "ci cd": "ci_cd",
    "aws": "aws",

    "android": "android",
    "firebase": "firebase",

    "algorithms": "algorithms",
    "algoritms": "algorithms",
    "data structure": "data_structures",
    "data structures": "data_structures",
    "competitive programming": "competitive_programming",

    "ui/ux": "ui_ux",
    "ui ux": "ui_ux",
    "figma": "figma",
}


RESUMES = [
    {
        "id": "01",
        "name": "Arjun Sharma",
        "skills": "Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning",
    },
    {
        "id": "02",
        "name": "Priya Nair",
        "skills": "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS",
    },
    {
        "id": "03",
        "name": "Rahul Gupta",
        "skills": "Java, Spring Boot, MySql, Microservices, Docker, kubernates",
    },
    {
        "id": "04",
        "name": "Sneha Patel",
        "skills": "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib",
    },
    {
        "id": "05",
        "name": "Vikram Singh",
        "skills": "C++, Algoritms, Data Structure, competitive programming, python",
    },
    {
        "id": "06",
        "name": "Ananya Krishnan",
        "skills": "javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD",
    },
    {
        "id": "07",
        "name": "Karan Mehta",
        "skills": "Python, Sklearn, XGboost, feature engineering, SQL, tableau",
    },
    {
        "id": "08",
        "name": "Deepika Rao",
        "skills": "Java, Android, Kotlin, Firebase, REST, UI/UX, figma",
    },
    {
        "id": "09",
        "name": "Aditya Kumar",
        "skills": "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest",
    },
    {
        "id": "10",
        "name": "Meera Iyer",
        "skills": "python, R, statistics, ML, regression, clustering, Power-BI",
    },
]


JOBS = [
    {
        "id": "JD-1",
        "company": "Kakao",
        "location": "Seoul",
        "role": "ML Engineer",
        "required": (
            "Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, SQL, "
            "Data Visualization"
        ),
        "preferred": "NLP, BERT, Feature Engineering, Statistics",
    },
    {
        "id": "JD-2",
        "company": "Naver",
        "location": "Seongnam",
        "role": "Backend Engineer",
        "required": (
            "Java, Spring Boot, MySQL, PostgreSQL, Microservices, Docker, "
            "Kubernetes"
        ),
        "preferred": "REST API, CI/CD, Redis",
    },
    {
        "id": "JD-3",
        "company": "Line",
        "location": "Seoul",
        "role": "Frontend Engineer",
        "required": "JavaScript, React, Vue, TypeScript, REST API, HTML/CSS",
        "preferred": "Node.js, GraphQL, Redux, Jest, AWS",
    },
]
