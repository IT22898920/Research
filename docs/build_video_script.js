const fs = require("fs");
const {
  Document,
  Packer,
  Paragraph,
  TextRun,
  AlignmentType,
  LevelFormat,
  BorderStyle,
} = require("docx");

const p = (children, opts = {}) =>
  new Paragraph({
    spacing: { after: 100, line: 280 },
    ...opts,
    children: Array.isArray(children) ? children : [children],
  });

const t = (text, opts = {}) => new TextRun({ text, ...opts });

const heading = (text, opts = {}) =>
  new Paragraph({
    spacing: { before: 200, after: 120, line: 280 },
    border: {
      bottom: {
        color: "888888",
        space: 4,
        style: BorderStyle.SINGLE,
        size: 6,
      },
    },
    ...opts,
    children: [t(text, { bold: true, size: 26 })],
  });

const subheading = (text) =>
  new Paragraph({
    spacing: { before: 160, after: 80, line: 280 },
    children: [t(text, { bold: true, size: 24, color: "1F4E79" })],
  });

const quoteBlock = (lines) =>
  lines.map((line, i) =>
    new Paragraph({
      spacing: { after: 80, line: 300 },
      indent: { left: 360, right: 360 },
      children: [
        t(line, {
          italics: true,
          size: 22,
        }),
      ],
    })
  );

const doc = new Document({
  styles: {
    default: {
      document: { run: { font: "Calibri", size: 22 } }, // 11pt
    },
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
          margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 },
        },
      },
      children: [
        // Title
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 80 },
          children: [
            t("CocoSense – CRI Expert Video Recommendation Script", {
              bold: true,
              size: 30,
            }),
          ],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 240 },
          children: [
            t(
              "For the Coconut Research Institute of Sri Lanka, Lunuwila",
              { italics: true, size: 22, color: "555555" }
            ),
          ],
        }),

        // Project info block
        subheading("Project & Team Reference"),
        new Paragraph({
          spacing: { after: 60, line: 280 },
          children: [
            t("Project Name: ", { bold: true }),
            t("CocoSense – An AI-Powered Drone-Based System for Comprehensive Coconut Tree Health Monitoring and Yield Prediction"),
          ],
        }),
        new Paragraph({
          spacing: { after: 60, line: 280 },
          children: [
            t("Mobile Application: ", { bold: true }),
            t("CocoSense (developed by SLIIT, Faculty of Computing)"),
          ],
        }),
        new Paragraph({
          spacing: { after: 60, line: 280 },
          children: [t("Research Team:", { bold: true })],
        }),
        new Paragraph({
          numbering: { reference: "bullets", level: 0 },
          spacing: { after: 40, line: 280 },
          children: [t("Pasanjith W.A.R. – IT22898920 – Drone-Based Pest Detection using Deep Learning")],
        }),
        new Paragraph({
          numbering: { reference: "bullets", level: 0 },
          spacing: { after: 40, line: 280 },
          children: [t("Panditharathne R.L. – IT22266200 – Drone-Based Coconut Yield Prediction")],
        }),
        new Paragraph({
          numbering: { reference: "bullets", level: 0 },
          spacing: { after: 40, line: 280 },
          children: [t("Subasinghe S.M.U. – IT22186638 – Drone-Based Coconut Tree Health Monitoring Using Leaf and Branch Images")],
        }),
        new Paragraph({
          numbering: { reference: "bullets", level: 0 },
          spacing: { after: 200, line: 280 },
          children: [t("Karunarathne B.G.T.N. – IT22027788 – Drone-Based Coconut Leaf Disease Detection using Deep Learning")],
        }),

        // Version 1 - Short
        heading("Version 1 — Short (≈ 25–30 seconds)"),
        new Paragraph({
          spacing: { after: 100 },
          children: [
            t("Best when the expert prefers a brief endorsement.", {
              italics: true,
              color: "555555",
            }),
          ],
        }),
        ...quoteBlock([
          "“Hello. I am [Name], [Designation] at the Coconut Research Institute of Sri Lanka, Lunuwila.",
          "I am pleased to recommend the research project ‘CocoSense’ – an AI-powered system for coconut tree health monitoring and yield prediction, developed by a group of final-year students from SLIIT: Pasanjith, Panditharathne, Subasinghe and Karunarathne.",
          "We consider CocoSense a timely and practical solution for the Sri Lankan coconut industry, and we are pleased to endorse this research. Thank you.”",
        ]),

        // Version 2 - Standard
        heading("Version 2 — Standard (≈ 45–60 seconds)  ⭐ Recommended"),
        new Paragraph({
          spacing: { after: 100 },
          children: [
            t("Balanced length – mentions students, app name, components and benefits.", {
              italics: true,
              color: "555555",
            }),
          ],
        }),
        ...quoteBlock([
          "“Hello. I am [Name], [Designation] at the Coconut Research Institute of Sri Lanka, Lunuwila.",
          "I am pleased to recommend the research project ‘CocoSense’ – an AI-powered drone-based system for coconut tree health monitoring and yield prediction. This research has been carried out by four final-year students of the Faculty of Computing at SLIIT: Pasanjith W.A.R., Panditharathne R.L., Subasinghe S.M.U., and Karunarathne B.G.T.N.",
          "The CocoSense mobile application uses Artificial Intelligence to detect coconut pests such as the Coconut Mite, Caterpillar and White Fly, identify leaf and bunch diseases, monitor overall tree health, and predict yield at the individual-tree level. It is supported by a custom IoT GPS device that helps map every tree accurately within a plantation.",
          "We at the Coconut Research Institute consider CocoSense a timely, practical and very valuable solution for the Sri Lankan coconut industry. We are pleased to endorse this work and we wish the team every success. Thank you.”",
        ]),

        // Version 3 - Long
        heading("Version 3 — Long (≈ 80–90 seconds)"),
        new Paragraph({
          spacing: { after: 100 },
          children: [
            t("Use only if the expert is willing to do a fuller statement.", {
              italics: true,
              color: "555555",
            }),
          ],
        }),
        ...quoteBlock([
          "“Hello. I am [Name], [Designation] at the Coconut Research Institute of Sri Lanka, in Lunuwila.",
          "Today I would like to introduce and recommend a research project titled ‘CocoSense – An AI-Powered Drone-Based System for Comprehensive Coconut Tree Health Monitoring and Yield Prediction.’ This work has been developed by four final-year undergraduate students of the Faculty of Computing at the Sri Lanka Institute of Information Technology, SLIIT – Pasanjith W.A.R., Panditharathne R.L., Subasinghe S.M.U., and Karunarathne B.G.T.N.",
          "The CocoSense mobile application combines deep-learning models, a backend analytics platform, and a custom IoT GPS device to support coconut growers in several important ways. It can detect major coconut pests including the Coconut Mite, Caterpillar and White Fly; identify leaf and bunch diseases; assess the overall health of the leaves, branches and tree; and estimate the yield of each individual tree in a plantation.",
          "The Sri Lankan coconut industry is currently facing serious challenges from pest infestations, climatic changes and the difficulty of inspecting plantations tree by tree. CocoSense provides a timely, affordable and field-ready solution that can be used directly by farmers and field officers, contributing to early detection, reduced crop loss, more accurate yield forecasting and stronger livelihoods for our growers.",
          "On behalf of the Coconut Research Institute of Sri Lanka, I am pleased to endorse the CocoSense research project as a valuable contribution to our national coconut sector, and to wish the SLIIT team every success in completing their research. Thank you.”",
        ]),

        // Version 4 - Comprehensive (full details)
        heading("Version 4 — Comprehensive (≈ 2 – 2½ minutes)  ★ Full details"),
        new Paragraph({
          spacing: { after: 100 },
          children: [
            t(
              "Covers every component, each student’s contribution, the AI models, the IoT device, cross-validation logic, and benefits. Use this if the expert is happy to give a longer, fuller endorsement.",
              { italics: true, color: "555555" }
            ),
          ],
        }),
        ...quoteBlock([
          "“Hello. I am [Name], [Designation] at the Coconut Research Institute of Sri Lanka, located in Lunuwila.",
          "It is my pleasure to introduce and recommend a final-year undergraduate research project titled ‘CocoSense – An AI-Powered Drone-Based System for Comprehensive Coconut Tree Health Monitoring and Yield Prediction.’ This project has been developed by four final-year students of the Faculty of Computing at the Sri Lanka Institute of Information Technology, SLIIT, working as a single research team:",
          "Pasanjith W.A.R., who is responsible for Drone-Based Pest Detection using Deep Learning;",
          "Panditharathne R.L., who has developed the Drone-Based Coconut Yield Prediction component;",
          "Subasinghe S.M.U., who has built the Drone-Based Coconut Tree Health Monitoring system using leaf and branch images;",
          "and Karunarathne B.G.T.N., who has developed the Drone-Based Coconut Leaf Disease Detection module.",
          "The students presented their work at our Institute and demonstrated its main components. CocoSense is built around a React Native mobile application, supported by a Flask-based AI inference service, a Node.js and MongoDB backend with analytics dashboards, and a custom IoT GPS device built on an ESP32 microcontroller and a NEO-M8N receiver, which is capable of mapping every individual coconut tree with approximately two-metre accuracy.",
          "The system uses several deep-learning models – including transfer-learning architectures such as EfficientNet and MobileNet – to detect the Coconut Mite, Caterpillar and White Fly infestations, identify leaf and bunch diseases such as Leaf Rot and Leaf Spot, assess overall leaf, branch and tree health, and estimate yield at the level of each individual tree. A cross-validation mechanism between the different models reduces false positives, and the application even rejects images that are not of coconut trees, which is very important for field use. For leaf-health issues, the application also provides growers with practical reasons, symptoms and treatment suggestions, which is highly valuable for non-expert users.",
          "The Sri Lankan coconut industry is currently facing serious challenges – including pest infestations, leaf and bunch diseases, climatic stress, and the difficulty of monitoring large plantations tree by tree. CocoSense directly addresses these problems and offers a timely, affordable and grower-friendly solution. It will help with early detection of pests and diseases, reduced crop losses, more accurate yield forecasting, better-informed decisions on fertilizer, irrigation and pest control, and the empowerment of small-holder farmers through a tool they can use directly on their own mobile phones.",
          "On behalf of the Coconut Research Institute of Sri Lanka, I am pleased to endorse the CocoSense research project as a valuable and practical contribution to our national coconut sector. We extend our continued co-operation to the student team and to SLIIT, and we wish them every success in completing this research. Thank you.”",
        ]),

        // Tips
        heading("Recording Tips"),
        new Paragraph({
          numbering: { reference: "bullets", level: 0 },
          spacing: { after: 60, line: 280 },
          children: [t("Record in landscape (horizontal) mode in 1080p if possible.")],
        }),
        new Paragraph({
          numbering: { reference: "bullets", level: 0 },
          spacing: { after: 60, line: 280 },
          children: [t("Frame the expert from chest up; keep the CRI sign / building visible behind if possible.")],
        }),
        new Paragraph({
          numbering: { reference: "bullets", level: 0 },
          spacing: { after: 60, line: 280 },
          children: [t("Choose a quiet indoor location to avoid wind noise; use the phone’s closer microphone or a clip-on mic.")],
        }),
        new Paragraph({
          numbering: { reference: "bullets", level: 0 },
          spacing: { after: 60, line: 280 },
          children: [t("Hold the phone steady on a tripod / table; do not zoom in digitally.")],
        }),
        new Paragraph({
          numbering: { reference: "bullets", level: 0 },
          spacing: { after: 60, line: 280 },
          children: [t("Ask the expert to speak naturally and a little slowly; pronouncing all four student surnames clearly.")],
        }),
        new Paragraph({
          numbering: { reference: "bullets", level: 0 },
          spacing: { after: 60, line: 280 },
          children: [t("Record at least two takes – one short, one standard – so you have a choice during editing.")],
        }),
        new Paragraph({
          numbering: { reference: "bullets", level: 0 },
          spacing: { after: 60, line: 280 },
          children: [t("Also request a separate 5–10 second clip where the expert says only: “I recommend the CocoSense research project.” This is useful as a closing voice-cut.")],
        }),
        new Paragraph({
          numbering: { reference: "bullets", level: 0 },
          spacing: { after: 200, line: 280 },
          children: [t("After recording, kindly thank the expert and confirm written permission to use the clip in academic / research presentations.")],
        }),

        // Optional Sinhala intro
        heading("Optional – Short Sinhala Intro (if requested)"),
        ...quoteBlock([
          "“ආයුබෝවන්. මම [නම], පොල් පර්යේෂණ ආයතනය, ලුණුවිල. SLIIT විශ්ව විද්‍යාලයේ අවසන් වසරේ සිසුන් සතරදෙනෙකු විසින් සිදුකරන ලද ‘CocoSense’ පර්යේෂණය - පොල් ගස්වල සෞඛ්‍යය හා අස්වැන්න පරීක්ෂාව සඳහා නිර්මාණය කරන ලද බුද්ධිමත් පද්ධතිය - අපි අගය කරමු හා ශ්‍රී ලංකාවේ පොල් කර්මාන්තයට උපයෝගී වැදගත් මෙහෙවරක් ලෙස නිර්දේශ කරමු. ස්තූතියි.”",
        ]),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  const outPath =
    "d:/SLIIT/Reaserch Project/CoconutHealthMonitor/Research/docs/CRI_Video_Recommendation_Script.docx";
  fs.writeFileSync(outPath, buf);
  console.log("Wrote:", outPath, "(" + buf.length + " bytes)");
});
