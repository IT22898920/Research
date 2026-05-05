const fs = require("fs");
const {
  Document,
  Packer,
  Paragraph,
  TextRun,
  AlignmentType,
  LevelFormat,
} = require("docx");

const t = (text, opts = {}) => new TextRun({ text, ...opts });

const sectionHeading = (text) =>
  new Paragraph({
    spacing: { before: 240, after: 100, line: 280 },
    children: [t(text, { bold: true, size: 26, color: "1F4E79" })],
  });

const bullet = (children) =>
  new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 80, line: 280 },
    children: Array.isArray(children) ? children : [children],
  });

const quote = (text) =>
  new Paragraph({
    spacing: { before: 60, after: 160, line: 300 },
    indent: { left: 360, right: 360 },
    children: [t(text, { italics: true, size: 24 })],
  });

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Calibri", size: 22 } } },
  },
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [
          {
            level: 0,
            format: LevelFormat.BULLET,
            text: "•",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } },
          },
        ],
      },
    ],
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 },
          margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 },
        },
      },
      children: [
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 80 },
          children: [
            t("CocoSense — Video Recommendation (Point-by-Point)", {
              bold: true,
              size: 30,
            }),
          ],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 240 },
          children: [
            t("Coconut Research Institute of Sri Lanka, Lunuwila", {
              italics: true,
              size: 22,
              color: "555555",
            }),
          ],
        }),

        // 1. Intro
        sectionHeading("1. Introduction"),
        quote("“Hello, I am [Name], [Designation] at the Coconut Research Institute of Sri Lanka, Lunuwila.”"),

        // 2. Project Recommendation
        sectionHeading("2. Project Recommendation"),
        quote("“I am pleased to recommend the research project ‘CocoSense’ — an AI-powered drone-based system for coconut tree health monitoring and yield prediction.”"),

        // 3. Research Team
        sectionHeading("3. The Research Team (SLIIT — Faculty of Computing)"),
        bullet([t("Pasanjith W.A.R. ", { bold: true }), t("— Drone-Based Pest Detection using Deep Learning")]),
        bullet([t("Panditharathne R.L. ", { bold: true }), t("— Drone-Based Coconut Yield Prediction")]),
        bullet([t("Subasinghe S.M.U. ", { bold: true }), t("— Tree Health Monitoring (leaf and branch images)")]),
        bullet([t("Karunarathne B.G.T.N. ", { bold: true }), t("— Coconut Leaf Disease Detection")]),

        // 4. System Components
        sectionHeading("4. System Components"),
        bullet([t("React Native mobile application ", { bold: true }), t("for use in the field")]),
        bullet([t("Flask-based AI inference service", { bold: true })]),
        bullet([t("Node.js + MongoDB backend ", { bold: true }), t("with analytics dashboards")]),
        bullet([t("Custom IoT GPS device ", { bold: true }), t("(ESP32 + NEO-M8N) — maps every tree with approximately two-metre accuracy")]),

        // 5. AI Capabilities
        sectionHeading("5. AI Capabilities"),
        bullet(t("Detects coconut pests — Coconut Mite, Caterpillar, and White Fly")),
        bullet(t("Identifies leaf and bunch diseases such as Leaf Rot and Leaf Spot")),
        bullet(t("Assesses leaf, branch, and overall tree health")),
        bullet(t("Predicts yield at the individual-tree level")),
        bullet(t("Uses transfer-learning architectures (EfficientNet, MobileNet)")),
        bullet(t("Cross-validation mechanism between models reduces false positives")),
        bullet(t("Rejects non-coconut images — important for field use")),
        bullet(t("Provides growers with reasons, symptoms, and treatment suggestions for leaf-health issues")),

        // 6. Industry Relevance
        sectionHeading("6. Industry Relevance"),
        bullet(t("The Sri Lankan coconut industry faces pest infestations, leaf and bunch diseases, climatic stress, and the difficulty of monitoring large plantations tree by tree.")),
        bullet(t("CocoSense addresses these challenges directly with a timely, affordable, and grower-friendly solution.")),

        // 7. Benefits to Society
        sectionHeading("7. Benefits to Society"),
        bullet(t("Early detection of pests and diseases")),
        bullet(t("Reduced crop losses")),
        bullet(t("More accurate yield forecasting")),
        bullet(t("Better decisions on fertilizer, irrigation, and pest control")),
        bullet(t("Empowerment of small-holder farmers through an easy-to-use mobile tool")),

        // 8. Endorsement
        sectionHeading("8. Endorsement"),
        quote("“On behalf of the Coconut Research Institute of Sri Lanka, I am pleased to endorse the CocoSense research project as a valuable and practical contribution to our national coconut sector.”"),

        // 9. Closing
        sectionHeading("9. Closing"),
        quote("“We wish the SLIIT team every success in completing this research. Thank you.”"),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  const outPath =
    "d:/SLIIT/Reaserch Project/CoconutHealthMonitor/Research/docs/CRI_Video_Script_Points.docx";
  fs.writeFileSync(outPath, buf);
  console.log("Wrote:", outPath, "(" + buf.length + " bytes)");
});
