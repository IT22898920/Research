const fs = require("fs");
const {
  Document,
  Packer,
  Paragraph,
  TextRun,
  AlignmentType,
} = require("docx");

const passage =
  "Hello. I am [Name], [Designation] at the Coconut Research Institute of Sri Lanka, located in Lunuwila. " +
  "It is my pleasure to introduce and recommend a final-year undergraduate research project titled “CocoSense – An AI-Powered Drone-Based System for Comprehensive Coconut Tree Health Monitoring and Yield Prediction,” developed by four final-year students of the Faculty of Computing at the Sri Lanka Institute of Information Technology, SLIIT — Pasanjith W.A.R., who is responsible for drone-based pest detection using deep learning; Panditharathne R.L., who has developed the coconut yield prediction component; Subasinghe S.M.U., who has built the coconut tree health monitoring system using leaf and branch images; and Karunarathne B.G.T.N., who has developed the coconut leaf disease detection module. " +
  "The students presented their work at our Institute, and CocoSense is built around a React Native mobile application supported by a Flask-based AI inference service, a Node.js and MongoDB backend with analytics dashboards, and a custom IoT GPS device built on an ESP32 microcontroller and a NEO-M8N receiver, capable of mapping every individual coconut tree with approximately two-metre accuracy. " +
  "The system uses several deep-learning models, including transfer-learning architectures such as EfficientNet and MobileNet, to detect Coconut Mite, Caterpillar and White Fly infestations, identify leaf and bunch diseases such as Leaf Rot and Leaf Spot, assess overall leaf, branch and tree health, and estimate yield at the level of each individual tree, while a cross-validation mechanism between the different models reduces false positives and the application even rejects images that are not of coconut trees, which is very important for field use; for leaf-health issues, the application also provides growers with practical reasons, symptoms and treatment suggestions, which is highly valuable for non-expert users. " +
  "The Sri Lankan coconut industry is currently facing serious challenges including pest infestations, leaf and bunch diseases, climatic stress, and the difficulty of monitoring large plantations tree by tree, and CocoSense directly addresses these problems by offering a timely, affordable and grower-friendly solution that supports early detection of pests and diseases, reduced crop losses, more accurate yield forecasting, better-informed decisions on fertilizer, irrigation and pest control, and the empowerment of small-holder farmers through a tool they can use directly on their own mobile phones. " +
  "On behalf of the Coconut Research Institute of Sri Lanka, I am pleased to endorse the CocoSense research project as a valuable and practical contribution to our national coconut sector, and we wish the SLIIT team every success in completing this research. Thank you.";

const doc = new Document({
  styles: {
    default: {
      document: { run: { font: "Times New Roman", size: 26 } }, // 13pt - easy to read
    },
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 }, // A4
          margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 },
        },
      },
      children: [
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 80 },
          children: [
            new TextRun({
              text: "CocoSense — Video Recommendation Script",
              bold: true,
              size: 30,
            }),
          ],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 320 },
          children: [
            new TextRun({
              text: "(Coconut Research Institute of Sri Lanka, Lunuwila)",
              italics: true,
              size: 22,
              color: "555555",
            }),
          ],
        }),
        new Paragraph({
          alignment: AlignmentType.JUSTIFIED,
          spacing: { after: 0, line: 360 }, // 1.5 line spacing for readability
          children: [new TextRun({ text: passage })],
        }),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  const outPath =
    "d:/SLIIT/Reaserch Project/CoconutHealthMonitor/Research/docs/CRI_Video_Script_SinglePassage.docx";
  fs.writeFileSync(outPath, buf);
  console.log("Wrote:", outPath, "(" + buf.length + " bytes)");
});
