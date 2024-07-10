# Distributed-Batch-Anomaly-Detection-in-Communication-Networks

## Abstract
Preventing security breaches using existing technologies alone is unrealistic. Therefore, intrusion detection systems (IDSs) play a crucial role in network security. Many current IDSs rely on rule-based mechanisms, which struggle with novel intrusion types and require time-consuming rule encoding based on known threats. This paper proposes innovative frameworks leveraging the random forests data mining algorithm across misuse, anomaly, and hybrid network-based IDSs.

Misuse Detection: The random forests algorithm constructs intrusion patterns from training data to detect intrusions by matching them against network activities.

Anomaly Detection: The algorithm's outlier detection identifies novel threats by detecting deviations from established network service patterns.

Hybrid System: Integrates the strengths of both misuse and anomaly detection methods to enhance detection capabilities.

Testing was conducted using the KDD '99 dataset. Results show:

The misuse detection method outperforms the best KDD '99 results.
The anomaly detection method achieves a higher detection rate with a lower false positive rate compared to other unsupervised approaches.
The hybrid system significantly boosts overall IDS performance.
__________________________
## Introduction

The increasing reliance on interconnected networks in personal and professional settings necessitates robust security measures. Network anomaly detection (NAD) plays a critical role in safeguarding these networks by identifying unusual patterns in network traffic that deviate from established baselines. These anomalies can signal potential security threats such as malware intrusions, denial-of-service attacks, or unauthorized access attempts.

Traditional NAD approaches often rely on signature-based detection, which matches network traffic patterns against known malicious signatures. However, this method struggles to keep pace with evolving cybercriminal tactics, which continuously develop new and sophisticated attack methods.

Machine learning (ML) offers a promising alternative for real-time NAD. ML algorithms can learn from historical network traffic data to identify normal behavior patterns and detect deviations that might indicate anomalies. This data-driven approach allows for a more adaptable and robust NAD system compared to signature-based methods.

Real-time processing of network traffic data further enhances the effectiveness of ML-based NAD by enabling early detection and mitigation of potential security threats. Frameworks like Apache Spark Streaming provide capabilities to analyze high-volume network traffic data streams in real-time, making them well-suited for implementing real-time ML-based NAD systems.

This research project explores supervised and unsupervised machine learning techniques for real-time network anomaly detection using Apache Spark Streaming. It evaluates the effectiveness of various supervised learning algorithms (Decision Tree, Random Forest, Gradient Boost Tree, Naive Bayes, Logistic Regression) and the K-Means clustering algorithm for unsupervised anomaly detection. By comparing these approaches, the project aims to identify the most suitable techniques for real-time NAD in network security.
_____________________________________
## Proposed Model

This research introduces a real-time network anomaly detection model using Apache Spark Streaming and supervised machine learning algorithms. The model components include:

1. Data Preprocessing: Ensuring data suitability through tasks like handling missing values and scaling features.
2. Model Training: Utilizing supervised algorithms (e.g., Random Forest, Decision Tree) on NSL-KDD to differentiate normal from anomalous traffic.
3. Spark Streaming Integration: Enabling real-time processing of network traffic streams.
4. Anomaly Detection and Alert Generation: Triggering alerts upon anomaly detection for timely response.
____________
## System Implementation and Testing

Implementation

- Data Collection and Preprocessing

We utilized the KDD99 dataset, comprising labeled network traffic data with both normal and malicious activities.
- Data Storage

Data packets captured by Wireshark are stored in Hadoop Distributed File System (HDFS) for scalability and accessibility.
- Data Preprocessing

Hadoop MapReduce is employed for parallel processing tasks like cleaning, filtering, and feature extraction from raw data packets.
- Model Development

Apache Spark's machine learning capabilities were used to develop intrusion detection models. Algorithms tested included Random Forest, Gradient Boosting, and Logistic Regression.
- Model Evaluation

Models were evaluated using metrics such as accuracy, precision, recall, and F1-score. Cross-validation and hyperparameter tuning ensured robustness.
2. Testing

- Real-Time Analysis

Real-time data from Wireshark was transformed into CSV format using Scapy for preprocessing and analysis.
- Experiment Setup

The dataset was split into training and testing sets to train models and evaluate their performance.
- Results

Our models demonstrated high accuracy in detecting various network intrusions, with Apache Spark proving effective for real-time analysis.

![image](https://github.com/Arpit-77/Distributed-Batch-Anomaly-Detection-in-Communication-Networks/assets/139072905/a61b37f2-2416-499c-afa2-045c7253f248)

_______
## Results and Discussion

The proposed model effectively detects network anomalies using Apache Spark Streaming and supervised learning algorithms.

## Experimental Results

Evaluation metrics such as accuracy, precision, recall, and F1-score were used to assess model performance.

## Comparison with Prior Research Efforts

Our model integrates real-time analysis, Apache Spark Streaming, and diverse algorithm exploration, benchmarked against the KDD99 dataset.

## Conclusion

This research advances network security with scalable, real-time anomaly detection, bridging academic research with practical cybersecurity applications.


![image](https://github.com/Arpit-77/Distributed-Batch-Anomaly-Detection-in-Communication-Networks/assets/139072905/dc28a855-6cd2-4904-ba2c-22f623b36edc)

The code can be found [here](Code.py)
