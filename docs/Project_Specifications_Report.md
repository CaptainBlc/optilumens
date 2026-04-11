CMPE 491
Senior Project I
Project Specifications Report
Group Members
Ahmet Emir Ceylan - 34126149190
Batuhan Taşdemir - 17965022282
Furkan Cabbar – 15550102548
1. Introduction
1.1. Description
This project aims to develop an AI-based camera image enhancement system capable of
improving the visual quality of low-light, noisy, or low-resolution images. Unlike
traditional filtering techniques that rely on fixed rules, this system uses deep learning
models that learn directly from data. Through this adaptive structure, the model can
brighten dark environments, reduce visual noise, sharpen blurred regions, and correct
color imbalances automatically.
The system is designed to be both efficient and sustainable. By incorporating optimized
model architectures and energy-aware processing strategies, the project minimizes
resource consumption while maintaining high-quality output. Built on an open-source
foundation, the solution supports collaboration, long-term adaptability, and transparent
development. With its flexible and data-driven nature, the system targets various realworld applications, including smart city surveillance, autonomous driving, healthcare
imaging, environmental monitoring, and industrial quality control.
1.2. Constraints
During development, several constraints must be taken into account:
Technical Constraints
• Computational Cost: Training deep learning models requires high GPU power and
long training times, which increases development cost and limits experimentation
speed.
• Real-Time Performance Challenges: High-resolution images require optimized
processing. Without proper model compression or architectural optimization, latency
may occur.
• Hardware Compatibility: Different deployment environments (embedded systems,
servers, local machines) vary in performance, making universal optimization difficult.
• Dataset Limitations and Labeling Constraints: Since the system is designed for
image enhancement rather than classification, the project does not rely on manually
labeled datasets. Instead, it requires large paired or unpaired datasets consisting of
high-quality and low-quality image samples.
This introduces several important constraints:
o High-quality paired datasets are difficult to collect and may require controlled
environments.
o Unpaired datasets (used in unsupervised enhancement techniques) may
introduce inconsistencies in model learning.
o The absence of traditional labels reduces annotation cost, but makes it harder
to quantitatively measure ground-truth accuracy.
o Ensuring dataset variety (lighting, noise types, camera sensors, environments)
is challenging but necessary for preventing model bias and improving
generalization.
o Ethical and legal considerations arise when datasets contain real-world
imagery, requiring proper anonymization or publicly licensed datasets.
Environmental and Sustainability Constraints
• Training and fine-tuning models consume significant energy.
• Efficient architectures are needed to reduce environmental impact and operational
cost.
Legal and Ethical Constraints
• Image datasets must comply with data protection rules (e.g., GDPR).
• Any dataset containing people or identifiable scenes must undergo anonymization or
use licensed sources.
• Fairness concerns arise if the dataset overrepresents certain environments, lighting
conditions, or camera types.
Operational Constraints
• The system must integrate with existing camera pipelines without requiring major
hardware changes.
• Consistency must be maintained across different image types and environmental
conditions.
• Long-term maintainability depends on a modular, open-source structure and clear
documentation.
1.3. Professional and Ethical Issues
Developing a system that processes visual data introduces several professional and ethical
responsibilities:
• Data Privacy & Security:
All image data must be handled with strict confidentiality. Secure storage,
encrypted transmission, and controlled access are essential.
• Fairness & Bias Prevention:
The system must avoid producing biased or misleading outputs. Training data and
model behavior should be monitored to ensure fairness across diverse conditions.
• Transparency & Explainability:
Users should have clear information regarding how the model processes images
and what types of data are used. Transparency builds trust and supports ethical
deployment.
• Regulatory Compliance:
Adherence to international and national data protection laws, AI ethics guidelines,
and domain-specific regulations is required throughout development.
• Responsible Deployment:
In sensitive domains like healthcare or surveillance, the system should undergo
thorough testing to minimize the risk of misinterpretation or harm.
2. Requirements
Functional Requirements
• The system shall enhance images captured under low-light, noisy, or low-resolution
conditions.
• It shall automatically adjust brightness, sharpness, noise levels, and color balance.
• It shall support common image file formats (e.g., JPEG, PNG, TIFF).
• Users shall be able to process images individually or in batches.
• The system shall provide visibly improved image quality without requiring manual
tuning.
Non-Functional Requirements
• Performance: The system must operate efficiently without excessive resource
consumption.
• Compatibility: It must run on a variety of hardware configurations and operating
environments.
• Modularity: The architecture should support easy updates, improvements, and
component replacement.
• Security: User data must be processed with strong privacy protection and secure
handling.
• Maintainability: Documentation and code structure should support long-term
development efforts.
• Scalability: The system should allow the integration of improved or alternative AI
models in the future.
3. References
• Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
• Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image Quality
Assessment: From Error Visibility to Structural Similarity. IEEE Transactions on
Image Processing.
• European Commission. (2021). Ethics Guidelines for Trustworthy AI.
• ISO/IEC 23091-4:2020 – Image Coding Systems and Quality Evaluation Metrics.