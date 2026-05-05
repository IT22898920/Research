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
    spacing: { before: 220, after: 100, line: 280 },
    children: [t(text, { bold: true, size: 26, color: "1F4E79" })],
  });

const bullet = (children) =>
  new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 80, line: 280 },
    children: Array.isArray(children) ? children : [children],
  });

const para = (text, opts = {}) =>
  new Paragraph({
    spacing: { after: 120, line: 280 },
    ...opts,
    children: [t(text)],
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
        // Title
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 80 },
          children: [
            t("CocoSense — Video Recommendation Brief", {
              bold: true,
              size: 30,
            }),
          ],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 240 },
          children: [
            t("Suggested Topics for the Coconut Research Institute Expert", {
              italics: true,
              size: 22,
              color: "555555",
            }),
          ],
        }),

        // Polite intro request
        para(
          "Dear Sir / Madam, we humbly request a short video recommendation of approximately 1 to 2 minutes covering the following topics. You are most welcome to use your own words and your own order — the list below is only a guide of suggested points."
        ),

        // 1. Self-Introduction
        sectionHeading("1. Self-Introduction"),
        bullet(t("Your name and designation")),
        bullet(t("Your position at the Coconut Research Institute of Sri Lanka, Lunuwila")),

        // 2. Project Reference
        sectionHeading("2. Project Reference"),
        bullet([
          t("Mention the project name: "),
          t("“CocoSense”", { bold: true }),
        ]),
        bullet(
          t(
            "Brief description: an AI-powered drone-based system for coconut tree health monitoring and yield prediction"
          )
        ),
        bullet([
          t("A final-year research project from "),
          t("SLIIT, Faculty of Computing", { bold: true }),
        ]),

        // 3. Research Team
        sectionHeading("3. The Research Team (optional but appreciated)"),
        bullet([t("Pasanjith W.A.R. ", { bold: true }), t("— Pest Detection")]),
        bullet([t("Panditharathne R.L. ", { bold: true }), t("— Yield Prediction")]),
        bullet([t("Subasinghe S.M.U. ", { bold: true }), t("— Tree Health Monitoring (leaf and branch)")]),
        bullet([t("Karunarathne B.G.T.N. ", { bold: true }), t("— Leaf Disease Detection")]),

        // 4. System Capabilities
        sectionHeading("4. Brief Mention of System Capabilities"),
        bullet(t("Detection of pests — Coconut Mite, Caterpillar, White Fly")),
        bullet(t("Identification of leaf and bunch diseases")),
        bullet(t("Tree health monitoring and yield prediction at the individual-tree level")),
        bullet(t("Mobile application together with an IoT GPS device for accurate tree mapping")),

        // 5. Industry Relevance
        sectionHeading("5. Industry Relevance"),
        bullet(t("Why this is a timely solution for the Sri Lankan coconut industry")),
        bullet(t("Current challenges — pests, diseases, and the difficulty of plantation-scale monitoring")),

        // 6. Benefits
        sectionHeading("6. Benefits"),
        bullet(t("Early detection of pests and diseases")),
        bullet(t("Reduced crop losses")),
        bullet(t("Better yield forecasting")),
        bullet(t("Empowerment of small-holder farmers")),

        // 7. Closing Endorsement
        sectionHeading("7. Closing Endorsement"),
        bullet(t("A few words endorsing the project on behalf of the Coconut Research Institute")),
        bullet(t("Best wishes for the SLIIT student team")),

        // Thank you
        new Paragraph({
          spacing: { before: 240, after: 0, line: 280 },
          alignment: AlignmentType.CENTER,
          children: [
            t(
              "Thank you very much for your kind support and time.",
              { italics: true, size: 22, color: "555555" }
            ),
          ],
        }),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  const outPath =
    "d:/SLIIT/Reaserch Project/CoconutHealthMonitor/Research/docs/CRI_Video_Recommendation_Brief.docx";
  fs.writeFileSync(outPath, buf);
  console.log("Wrote:", outPath, "(" + buf.length + " bytes)");
});
