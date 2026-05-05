const fs = require("fs");
const {
  Document,
  Packer,
  Paragraph,
  TextRun,
  AlignmentType,
  LevelFormat,
} = require("docx");

const today = new Date();
const dateStr = today.toLocaleDateString("en-GB", {
  day: "2-digit",
  month: "long",
  year: "numeric",
});

const p = (children, opts = {}) =>
  new Paragraph({
    spacing: { after: 100, line: 260 },
    ...opts,
    children: Array.isArray(children) ? children : [children],
  });

const t = (text, opts = {}) => new TextRun({ text, ...opts });

const doc = new Document({
  styles: {
    default: {
      document: { run: { font: "Times New Roman", size: 22 } }, // 11pt
    },
  },
  numbering: {
    config: [
      {
        reference: "students",
        levels: [
          {
            level: 0,
            format: LevelFormat.DECIMAL,
            text: "%1.",
            alignment: AlignmentType.LEFT,
            style: {
              paragraph: { indent: { left: 720, hanging: 360 } },
            },
          },
        ],
      },
    ],
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 }, // A4
          margin: {
            top: 1080,
            right: 1080,
            bottom: 1080,
            left: 1080,
          }, // ~0.75 inch
        },
      },
      children: [
        // Letterhead - CRI
        p(
          t("COCONUT RESEARCH INSTITUTE OF SRI LANKA", {
            bold: true,
            size: 26,
          }),
          {
            alignment: AlignmentType.CENTER,
            spacing: { after: 40 },
          }
        ),
        p(t("Lunuwila, Sri Lanka", { size: 20 }), {
          alignment: AlignmentType.CENTER,
          spacing: { after: 40 },
        }),
        p(t("Office of the Director", { italics: true, size: 20 }), {
          alignment: AlignmentType.CENTER,
          spacing: { after: 240 },
        }),

        // Date
        p(t(`Date: ${dateStr}`), { spacing: { after: 200 } }),

        // Recipient (Prof. Koliya Pulasinghe at SLIIT)
        p(t("Prof. Koliya Pulasinghe,")),
        p(t("Lecturer-in-Charge / Research Coordinator,")),
        p(t("Faculty of Computing,")),
        p(t("Sri Lanka Institute of Information Technology (SLIIT),")),
        p(t("New Kandy Road, Malabe."), { spacing: { after: 200 } }),

        // Salutation
        p(t("Dear Prof. Pulasinghe,"), { spacing: { after: 160 } }),

        // Subject
        p(
          t(
            "LETTER OF RECOMMENDATION – CocoSense RESEARCH PROJECT",
            { bold: true, underline: {} }
          ),
          { alignment: AlignmentType.CENTER, spacing: { after: 200 } }
        ),

        // Intro - CRI's perspective
        p(
          t(
            "I am pleased to write this letter on behalf of the Coconut Research Institute of Sri Lanka (CRISL), Lunuwila, with reference to the final-year undergraduate research project titled “CocoSense – An AI-Powered Drone-Based System for Comprehensive Coconut Tree Health Monitoring and Yield Prediction,” being conducted under your supervision by the following students of the Faculty of Computing, SLIIT:"
          )
        ),

        // Student list
        new Paragraph({
          numbering: { reference: "students", level: 0 },
          spacing: { after: 60, line: 260 },
          children: [
            t("Pasanjith W.A.R. – IT22898920 – "),
            t("Drone-Based Pest Detection using Deep Learning", { italics: true }),
          ],
        }),
        new Paragraph({
          numbering: { reference: "students", level: 0 },
          spacing: { after: 60, line: 260 },
          children: [
            t("Panditharathne R.L. – IT22266200 – "),
            t("Drone-Based Coconut Yield Prediction", { italics: true }),
          ],
        }),
        new Paragraph({
          numbering: { reference: "students", level: 0 },
          spacing: { after: 60, line: 260 },
          children: [
            t("Subasinghe S.M.U. – IT22186638 – "),
            t(
              "Drone-Based Coconut Tree Health Monitoring Using Leaf and Branch Images",
              { italics: true }
            ),
          ],
        }),
        new Paragraph({
          numbering: { reference: "students", level: 0 },
          spacing: { after: 200, line: 260 },
          children: [
            t("Karunarathne B.G.T.N. – IT22027788 – "),
            t("Drone-Based Coconut Leaf Disease Detection using Deep Learning", {
              italics: true,
            }),
          ],
        }),

        // System overview & components
        p(
          t(
            "The students presented their work at our Institute and demonstrated the major components of the proposed system, namely: a React Native mobile application for field officers and growers, a suite of TensorFlow-based deep-learning models for the detection of coconut pests (Coconut Mite, Caterpillar and White Fly), leaf and bunch diseases, leaf health and overall tree health, a Flask-based AI inference service with cross-validation between models, a Node.js / MongoDB backend with analytics dashboards, and a custom-built ESP32 + NEO-M8N IoT GPS device that maps every tree with approximately two-metre accuracy."
          )
        ),

        // Importance / timeliness / good solution
        p(
          t(
            "The coconut industry is one of the most important agricultural sectors in Sri Lanka, and it is currently facing serious challenges from pest infestations, leaf and bunch diseases and the difficulty of monitoring large plantations tree by tree. We consider CocoSense to be a highly timely and practically relevant solution that effectively combines Artificial Intelligence, mobile technology and IoT hardware to address these challenges in an affordable, scalable and grower-friendly manner."
          )
        ),

        // Suggestions + benefits to society
        p(
          t(
            "After reviewing the project, our experts have provided technical suggestions to further strengthen the scientific validity and field applicability of the system, particularly regarding pest classification, disease symptom interpretation and yield estimation under Sri Lankan plantation conditions. The Institute is satisfied that the project, once completed, will deliver clear benefits to the coconut sector and to society at large – including early detection of pests and diseases, reduced crop losses, more accurate yield forecasting, better-informed farmer decisions and the empowerment of small-holder growers through an easy-to-use mobile platform."
          )
        ),

        // Endorsement
        p(
          t(
            "On behalf of the Coconut Research Institute of Sri Lanka, I am therefore pleased to recommend and endorse the CocoSense research project as a valuable and timely contribution to the Sri Lankan coconut industry, and to extend our continued co-operation to the student team and to SLIIT throughout the remainder of the research."
          )
        ),

        p(t("Thank you for your kind consideration."), { spacing: { after: 240 } }),

        // Sign-off (Director)
        p(t("Yours sincerely,"), { spacing: { after: 480 } }),
        p(t("……………………………………………………………………"), {
          spacing: { after: 40 },
        }),
        p(t("Director", { bold: true }), { spacing: { after: 40 } }),
        p(t("Coconut Research Institute of Sri Lanka,"), { spacing: { after: 40 } }),
        p(t("Lunuwila."), { spacing: { after: 40 } }),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  const outPath =
    "d:/SLIIT/Reaserch Project/CoconutHealthMonitor/Research/docs/CRI_Recommendation_Letter_to_SLIIT.docx";
  fs.writeFileSync(outPath, buf);
  console.log("Wrote:", outPath, "(" + buf.length + " bytes)");
});
