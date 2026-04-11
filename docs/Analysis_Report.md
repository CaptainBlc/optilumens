CMPE 491
Senior Project I
Analysis Report







Group Members
Ahmet Emir Ceylan - 34126149190
Batuhan Taşdemir - 17965022282
Furkan Cabbar – 15550102548


1. Introduction
In today’s digital environment, image enhancement has become a critical requirement across sectors such as security, healthcare, consumer electronics, and digital media. Yet, a major limitation persists: most existing enhancement tools are highly dependent on the source of the image. Factors such as camera model, lens quality, sensor type, resolution, and lighting conditions significantly affect the performance of current enhancement solutions. As a result, images captured by low-quality or mismatched devices—such as surveillance cameras, drones, or outdated smartphones—often fail to be enhanced reliably or consistently.
The objective of this project is to develop a hybrid, device-independent image enhancement system capable of processing any input image, regardless of its origin, camera type, or baseline quality. The proposed system integrates three complementary enhancement methods—label-based enhancement, pixel-level processing, and general/global image improvement—into a unified architecture. Unlike AI-dependent tools that function as “black boxes” and may hallucinate unrealistic details, this system aims to provide a transparent, controlled, and explainable enhancement pipeline based on multi-layer image understanding.
2. Current System 
Existing image enhancement systems are typically constrained by their training datasets or by predefined assumptions about camera characteristics. For instance, enhancement models trained primarily on smartphone datasets tend to perform poorly on CCTV footage, medical imaging, or industrial camera outputs. Similarly, tools optimized for high-resolution images struggle with low-light, blurry, or pixel-degraded images.
Most current systems rely on only one enhancement strategy, such as:
•	Pixel-level methods (noise reduction, sharpening, denoising filters),
•	Model-based methods (semantic segmentation, region detection), or
•	Global enhancement (color balancing, contrast adjustments).
This single-method dependency reduces adaptability and leads to inconsistent or unstable results across diverse image types.To overcome this limitation, the proposed system adopts a multi-method hybrid pipeline, ensuring robustness regardless of device, resolution, or environmental constraints.
3. Proposed System
3.1 Overview
The proposed solution is a hybrid image enhancement platform designed to operate independently of the camera, device type, resolution, or lighting conditions. It integrates three complementary enhancement approaches:

Label-Based Processing
Uses semantic labels, masks, or region detection to selectively enhance specific areas of the image—such as faces, text, objects, edges, or structural surfaces—while avoiding distortion or over-processing of irrelevant regions.
Pixel-Level Processing
Applies noise reduction, deblurring, sharpening, and fine-grained detail restoration directly at the pixel matrix level, ensuring clarity and fidelity.
General Image Enhancement
Performs global improvements such as exposure correction, contrast balancing, color normalization, and illumination optimization.
By combining these three methods within a single pipeline, the system achieves highly controlled, explainable, and device-agnostic image enhancement without depending on any specific camera model or dataset.
3.2 Functional Requirements
•	The system shall accept image inputs from any camera or device type, including mobile phones, drones, CCTV systems, and webcams.
•	Users shall be able to upload an image or capture one in real time through a connected camera (optional in early prototypes).
•	The system shall automatically analyze the input image and determine which enhancement techniques are required.
•	The system shall apply all three enhancement approaches (label-based, pixel-level, and general enhancement).
•	The system shall display both the original and enhanced images using a before-and-after comparison interface.
•	Users shall be able to save or download the enhanced image.
•	Core enhancement operations shall remain functional without an internet connection (offline support).
3.3 Nonfunctional Requirements
•	The system should operate efficiently even on mid-range devices with limited computational power.
•	The user interface should remain simple, intuitive, and easy to use.
•	The average processing time per image should not exceed 10 seconds on standard consumer devices.
•	The system should minimize battery usage for mobile platforms.
•	User images and data must be handled securely and never stored or shared without explicit consent.
•	The application should be compatible across Android, iOS, and Windows platforms.

3.4 Pseudo Requirements
•	The system must utilize at least two Python-based image processing libraries such as OpenCV, PIL, scikit-image, or equivalents.
•	The user interface must support drag-and-drop image uploads.
•	If AI models are required, PyTorch or TensorFlow shall be the preferred frameworks.
•	The first prototype may be limited to processing uploaded images (real-time processing optional in later phases).
•	A “before-after” comparison view is mandatory for demonstration and evaluation purposes.
3.5 System Models
To formally describe how the system functions, multiple system modeling techniques will be used, including scenario descriptions, use case diagrams, class and object models, dynamic models, and user interface mock-ups.
The following system models describe how users interact with the system, how data flows through the enhancement pipeline, and how internal components collaborate to generate the final enhanced output. These models provide a structured, formal representation of the proposed hybrid enhancement system.
3.5.1 Scenarios
 
Scenario 1: Uploading and Enhancing a Low-Quality Image
1.	The user opens the application and navigates to the enhancement interface.
2.	The user drags and drops an image or selects one from their device.
3.	The system analyzes the image to determine its characteristics (noise level, blur, lighting conditions).
4.	The system selects the appropriate enhancement pipeline (label-based, pixel-level, global enhancement, or all three).
5.	The system processes the image and generates an enhanced version.
6.	The user views the before-and-after comparison.
7.	The user downloads or saves the enhanced image.
Scenario 2: Selective Region Enhancement (e.g., face, text, objects)
1.	The user uploads an image containing specific objects (e.g., a face, license plate, text).
2.	The system performs semantic segmentation to identify regions of interest.
3.	The system applies label-based enhancements only to relevant regions.
4.	Pixel-level and global enhancements are applied to the entire image.
5.	The system merges all enhancement layers into a unified output.
6.	The user reviews the result and downloads the image.
Scenario 3: Offline Enhancement on a Low-End Device
1.	The user launches the offline version of the application.
2.	The user uploads a low-resolution image from a basic smartphone.
3.	The image is processed using lightweight pixel-level and global filters.
4.	Label-based processing is executed using a compressed on-device model (if available).
5.	The system displays the enhanced output with minimal computational overhead.
6.	The user saves the image locally.
Scenario 4: Real-Time Image Capture (Future Extension)
1.	The user opens the camera interface in the application.
2.	The user captures an image through the device camera.
3.	The captured image is immediately analyzed and enhanced.
4.	Enhanced results appear within a short response time.
5.	The user accepts or discards the enhanced image.
3.5.2 Use Case Model

             
1.Use Case Name: Upload Image
Actor	User
Description	The user selects or captures an image from any camera source (smartphone, CCTV, drone, webcam, etc.) and uploads it into the system for enhancement.
Trigger	The user clicks the “Upload Image” button or drags-and-drops a file into the upload area.
Pre-conditions	The user must have access to the application.
A valid image file must be available on the device or through the connected camera.
Post-conditions	The selected image is successfully uploaded.
The image is temporarily stored in the system’s memory for analysis.
Main Flow	1.	The user initiates the upload process.
2.	The system opens the file selection dialog or camera capture interface.
3.	The user selects or captures an image.
4.	The system validates file format and resolution.
5.	The system confirms successful upload.
Alternative Flow	If the upload fails due to unsupported format, corrupted file, or connection issues, the system displays an error message and allows the user to retry.



2.Use Case Name: Process Image

Actor	User, System
Description	The system analyzes the uploaded image and enhances it using all three processing methods:
1.	Label-based enhancement,
2.	Pixel-level enhancement,
3.	General/global enhancement.
Trigger	The enhancement process begins automatically after upload, or
The user manually clicks “Enhance Image”.
Pre-conditions	A valid image must be uploaded and available in the system.
Post-conditions	The enhanced image is generated.
Both original and enhanced versions are stored temporarily for comparison.
Main Flow	1.	The system analyzes the image for noise, blur, lighting, and detectable objects.
2.	The system applies label-based enhancement to recognized regions.
3.	The system performs pixel-level operations (denoising, sharpening, deblurring).
4.	The system applies global enhancement (contrast, exposure, color correction).
5.	The system saves the enhanced output in memory.
Alternative Flow	If detailed analysis fails, the system falls back to default enhancement settings.
If a module fails, the system continues with available modules to ensure output.







3. Use Case Name: View Before/After Comparison
Actor	User
Description	The user views the original and enhanced images side-by-side or through a slider interface in order to assess image quality improvements.
Trigger	Image enhancement has been completed successfully.
Pre-conditions	Both original and enhanced images must exist in temporary storage.
Post-conditions	The user visually evaluates the results and chooses whether to download or reprocess the image.
Main Flow	1.	The system presents both images in a comparison interface.
2.	The user views and evaluates differences using slider or split-screen mode.
3.	The user decides whether to download, re-enhance, or discard the image.
Alternative Flow	If the user is not satisfied with the enhancement, they request reprocessing with different settings.

4.Use Case Name: Download Enhanced Image
Actor	User
Description	The user saves the enhanced image to their device in the preferred format and quality.
Trigger	The user clicks “Download” or “Save Image”.
Pre-conditions	The enhanced image must already be generated.
Post-conditions	The image file is saved to the user’s local device.
Temporary processing data may be cleared afterward.
Main Flow	1.	The user selects the download option.
2.	The system prepares the enhanced image file (format, resolution, compression).
3.	The user downloads or saves the file to their device.
Alternative Flow	If download fails due to file system or network issues,
the system offers retry options or alternative export formats.






3.5.3 Object and Class Model
Below is the conceptual class model describing internal components and their relationships.
 
1. User
Represents the end-user interacting with the system.
Methods:
•	uploadImage() – Initiates the process of selecting and uploading an image.
•	downloadImage() – Downloads the enhanced image to the device.
2. Image
Stores the raw image provided by the user and handles basic I/O operations.
Attributes:
•	rawData – Binary matrix of the original image.
•	resolution – Image resolution in pixels.
•	format – File format (JPEG, PNG, etc.)
Methods:
•	load() – Loads an image into the system.
•	save() – Saves processed images to local storage.

3. ImageProcessor
The central controller class responsible for orchestrating the enhancement pipeline.
Methods:
•	analyzeImage() – Evaluates lighting, noise, blur, and detectable patterns.
•	enhanceImage() – Coordinates all enhancement modules (label-based, pixel-level, and general enhancement).
Relationships:
•	Aggregates: Image, LabelBasedProcessor, PixelBasedProcessor, GeneralEnhancer.
4. LabelBasedProcessor
Handles object detection and region-specific enhancement.
Methods:
•	detectObjects() – Performs labeling or segmentation to identify faces, text, or key objects.
•	enhanceLabeledRegions() – Applies enhancement techniques only to the detected regions.

5. PixelBasedProcessor
Improves the overall pixel-level quality of the image.
Methods:
•	reduceNoise() – Removes noise and artifacts.
•	sharpenImage() – Restores edges and details.
6. GeneralEnhancer
Applies global image improvements.
Methods:
•	adjustBrightness() – Fixes underexposed or overexposed images.
•	improveContrast() – Enhances visibility of key features.
•	optimizeColors() – Balances saturation, hue, and white levels.



3.5.4 Dynamic Models
The following sequence model illustrates the end-to-end workflow of the enhancement system.
The description is fully aligned with the sequence diagram you shared.
 
Actor: User
System Components:
•	System
•	ImageProcessor
•	LabelBasedProcessor
•	PixelBasedProcessor
•	GeneralEnhancer
Main Flow (Textual UML Sequence Explanation)
1.	User → System:
The user uploads an image.
2.	System → ImageProcessor:
The system passes the raw image to ImageProcessor.
3.	ImageProcessor → ImageAnalyzer Logic:
analyzeImage() is executed to determine enhancement needs.
4.	ImageProcessor → LabelBasedProcessor:
detectObjects() is called to identify faces, text, or structural regions.
→ The system receives _labeledImage.
5.	ImageProcessor → PixelBasedProcessor:
reduceNoise() and sharpenImage() are applied.
→ Output: pixelEnhancedImage.
6.	ImageProcessor → GeneralEnhancer:
applyGeneralEnhancement() (brightness, contrast, color optimization).
→ Output: finalEnhancedImage.
7.	ImageProcessor → System:
The final enhanced image is returned to the system.
8.	System → User:
The processed image is displayed, and the user is given a download option.
Alternative Behaviors
•	If object detection fails
→ The system continues with pixel-based and general enhancement only.
•	If pixel enhancement fails
→ General enhancement is applied to ensure minimum output quality.
•	If any processor returns an error
→ The system notifies the user and uses fallback enhancement strategies.
3.5.5 User Interface
 
At this stage of the project, a graphical user interface (GUI) has not yet been implemented.
However, the system is expected to include the following UI elements in future development phases:
•	Image upload area (drag-and-drop or file selection)
•	Enhancement trigger button
•	Before/after comparison panel
•	Download button for the enhanced image
These UI components will be defined in detail once the frontend development phase begins.
Currently, the system models describe only backend processing components.
4. Glossary
Image Enhancement:
The process of improving the visual quality of an image by reducing noise, increasing sharpness, adjusting brightness/contrast, or making important details more visible.
Camera-Independent System:
A system that can process and enhance images without relying on any specific camera type, resolution, or hardware characteristics.
Label-Based Processing:
An enhancement approach where certain regions or objects in an image (such as faces, text, vehicles, or specific textures) are detected and processed differently based on their labels or categories.
Pixel-Level Processing:
A low-level image processing technique where operations are applied directly to individual pixels or small neighborhoods of pixels (e.g., noise reduction, sharpening, blur removal).
General Image Enhancement Methods:
A set of techniques that globally adjust properties of the image, such as brightness, contrast, color balance, and overall tone, without focusing on specific objects.
Noise:
Unwanted random variations in pixel values that degrade the quality of the image, often appearing as grain, speckles, or colored dots.
Blur:
A type of distortion where image details appear smeared or out of focus, usually caused by motion, defocus, or low-quality optics.
Resolution:
The amount of detail present in an image, typically expressed as the number of pixels in width and height (e.g., 1920×1080).
Dataset:
A structured collection of images used for training, testing, or evaluating the performance of image processing or machine learning algorithms.
Hybrid Processing:
The combination of multiple processing techniques or algorithms (e.g., label-based, pixel-level, and general enhancement) within a single system to achieve better results.
Processing Pipeline:
The ordered sequence of steps that an image passes through in the system, such as input → analysis → enhancement → output.
5. References
•	R. C. Gonzalez and R. E. Woods, Digital Image Processing, 4th Edition, Pearson, 2018.
•	K. Jain, Fundamentals of Digital Image Processing, Prentice Hall, 1989.
•	OpenCV Documentation, “Image Processing Module,” Available at: https://docs.opencv.org
•	(Accessed: 2025).
•	S. Schaefer and T. McPhail, “Image Deblurring and Denoising Techniques: A Survey,” Journal of Visual Communication and Image Representation, vol. XX, no. XX, pp. XX–XX, Year.
•	Python Software Foundation, “Python Imaging Library (Pillow) Documentation,” Available at: https://pillow.readthedocs.io (Accessed: 2025).
•	Any relevant course notes, tutorials, or technical blogs used during the development of this project.





