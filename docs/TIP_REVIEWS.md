Manuscript: TIP-34965-2025, "MCTTA: Multi-class Token Transformer Adapter for Weakly Supervised Semantic Segmentation".

Dear Dr. DINGJIE PENG,

I am writing to you concerning the above referenced manuscript, which you submitted to the IEEE Transactions on Image Processing (T-IP).

Based on the enclosed set of reviews, I regret to inform you that your manuscript has been rejected for publication.

This decision is based on several critical issues raised by the reviewers:
(1)    Insufficient novelty in the proposed method.
(2)    Lack of detailed explanations and insufficient clarity in some technical aspects.
(3)    Lack of comparison with more recent and competitive baseline methods.
(4)    Inadequate evidence to convincingly support the claims made in the paper.

Please note that according to IEEE Signal Processing Society policy "Handling of Rejected Papers" ([http://signalprocessingsociety.org/volunteers/policy-and-procedures-manual](http://signalprocessingsociety.org/volunteers/policy-and-procedures-manual)), the Society prohibits resubmission of rejected manuscripts more than once. Authors should carefully review the aforementioned policy before resubmitting their manuscript.

If you have any questions regarding the reviews, please contact me.  Any other inquiries should be directed to Patrick Gillespie.

Thank you for submitting your work to the IEEE Transactions on Image Processing. We hope you consider us again in the future.

Sincerely,

Dr. Jianwen Xie
Associate Editor
IEEE Transactions on Image Processing
[jianwen@ucla.edu](mailto\:jianwen@ucla.edu)  

Patrick Gillespie
Coordinator Society Publications
IEEE Signal Processing Society
[p.gillespie@ieee.org](mailto\:p.gillespie@ieee.org)

Reviewer Comments:

Reviewer: 1

Comments:
1\. Summary

This paper introduces MCTTA, a adapter-based framework for Weakly Supervised Semantic Segmentation (WSSS). The method aims to improve upon the MCTformer+ baseline by addressing several of its perceived architectural limitations. The core contributions include the integration of a hierarchical, graph-based feature extractor (the Spatial Prior Grapher) and several new modules designed to enhance feature fusion, improve training convergence, and refine Class Activation Maps (CAMs). The authors evaluate their method on the PASCAL VOC 2012 and MS COCO datasets, reporting SOTA performance in both single-stage and multi-stage WSSS settings.

2\. Major Concerns

While the proposed method is technically interesting and shows promising results, the manuscript has several major flaws in its narrative structure and experimental validation that prevent it from being considered for publication in its current form.

1\) Manuscript Structure and Narrative Flow Require a Overhaul
The paper’s current structure hinders readability and fails to build a compelling argument for its contributions.

Introduction: The introduction is not well-structured and needs a complete revision. I would strongly recommend the following structure: A single, concise opening paragraph defining the core problem (the high cost of FSSS vs. the motivation for WSSS). The current first paragraph is too long and should be condensed. A second paragraph that reviews existing methods and, based on their limitations, explicitly defines the research gap. A third, brief paragraph that introduces the proposed MCTTA framework as the solution. The content from the current paragraphs 3-6 should be merged and condensed into this single paragraph. This should be followed by the existing bulleted list of contributions.

Establishing the Research Gap and Contributions: The connection between the identified problems and the claimed contributions is weak. The contributions are presented as a list of technical modules and performance claims. For these to have impact, the introduction must first convincingly establish that the problems being solved (e.g., "lack of hierarchical feature extraction" ) are widely recognized and critical limitations in the WSSS field. The current introduction states these limitations but does not sufficiently substantiate them with a thorough analysis of prior art. This makes the contributions feel more like a set of technical improvements rather than solutions to a general, well-established research gap.

Redundancy: The manuscript contains redundant sections. The "Related Work" (Section II) and "Preliminaries" (Section III) should be merged into a single, cohesive "Background" section to create a more streamlined narrative.

2\) Experimental Comparisons Are Not Current
The primary weakness of the experimental evaluation is that the baseline comparisons are not up-to-date. For a manuscript under review in late 2025, it is essential to compare against the relevant state-of-the-art models from throughout 2024 and 2025. The current comparisons, while extensive, appear to stop with work from early 2024. Without these contemporary baselines, the claims of achieving SOTA performance cannot be fully substantiated. The authors must update their experiments to reflect the current research landscape in WSSS to accurately contextualize the contribution of MCTTA.

3\. Minor Concerns

1\) Figure Quality
Some components of the figures in the manuscript appear to be low-resolution bitmaps (CNN blocks). To meet publication standards, all figures should be provided in a vector graphics format (e.g., PDF, SVG).

4\. Recommendation

In summary, while the proposed MCTTA framework has novel components, the manuscript requires a revision of its structure, narrative, and a significant update to its experimental evaluation.

Therefore, my recommendation is Review Again After Major Changes.

Additional Questions:
Is the work within the scope of the journal?: Clearly within scope

Is the manuscript technically correct?: Some minor concerns that should be easily corrected without altering the contribution or conclusions

Is the technical contribution novel?: Moderate novelty, with clear extensions of existing methods/concepts

Is the technical contribution significant?: Moderate contribution, with the possibility of an impact on the field

Is the length of the manuscript appropriate to the contribution?: Length is appropriate

Are the references appropriate, without any significant omissions?: Complete list of references without any significant omissions

Are there any references that do not appear to be relevant?: All references are directly relevant to the contribution of the manuscript

Is the manuscript properly structured and clearly written?: Moderate issues of exposition that may require some time to correct, but do not substantially affect the ability to evaluate the technical content

If you are suggesting additional references they must be entered in the text box provided.  All suggestions must include full bibliographic information plus a DOI.
: N/A


Reviewer: 2

Comments:
This manuscript presents MCTTA, a novel framework that achieves impressive, state-of-the-art results in Weakly Supervised Semantic Segmentation (WSSS) across multiple standard benchmarks. The paper's strength lies in this significant empirical achievement, driven by a sophisticated architecture that integrates several effective modules like the Spatial Prior Grapher (SPG) and Class Token Projection (CTP). However, the work requires a major revision because it currently reads as a report of a successful engineering effort rather than a deep scientific contribution. The primary flaw is the insufficient methodological justification for its core architectural choices; for instance, the central claim about the benefit of a "graph-based" adapter is not substantiated with a crucial ablation study comparing it against a simpler, non-graph alternative.

Furthermore, the analysis of the results lacks the depth expected for a top-tier publication. The paper's success hinges on producing exceptionally high-quality Class Activation Maps (CAMs) , yet it fails to provide a deep, quantitative analysis of why these CAMs are superior beyond qualitative examples. Key findings, such as the dramatic performance gap between the direct method (MCTTA-D) and the baseline (MCTformer+-D), are presented without explanation or insight. To be suitable for publication, the authors must strengthen the manuscript's scientific rigor by providing more robust experimental justifications for their design choices and offering a more profound analysis of the mechanisms behind their model's success.


Additional Questions:
Is the work within the scope of the journal?: Clearly within scope

Is the manuscript technically correct?: Moderate concerns with the potential for some impact on the contribution or conclusions

Is the technical contribution novel?: Moderate novelty, with clear extensions of existing methods/concepts

Is the technical contribution significant?: Substantial contribution, with a clear potential for impact

Is the length of the manuscript appropriate to the contribution?: Length is appropriate

Are the references appropriate, without any significant omissions?: A largely complete list of references with only minor omissions that would not affect the novelty of the submission

Are there any references that do not appear to be relevant?: All references are directly relevant to the contribution of the manuscript

Is the manuscript properly structured and clearly written?: Moderate issues of exposition that may require some time to correct, but do not substantially affect the ability to evaluate the technical content

If you are suggesting additional references they must be entered in the text box provided.  All suggestions must include full bibliographic information plus a DOI.
: Zhang, B., Yu, S., Wei, Y., Zhao, Y., & Xiao, J. Frozen CLIP: A Strong Backbone for Weakly Supervised Semantic Segmentation. CVPR 2024. DOI: 10.1109/CVPR52733.2024.00364


Reviewer: 3

Comments:
Summary:
The manuscript proposed a Multi-Class Token Transformer Adapter (MCTTA) for Weakly Supervised Semantic Segmentation (WSSS). MCTTA incorporates Vision GNN into MCTformer+[1] using Hierarchical Fusion with improvements such as Class Token Projection (CTP), Split Weighted Softmax. MCTTA has shown competitive performance and even outperforms multiple state-of-the-art models in different paradigms on Pascal VOC 2012 and COCO 2014 datasets.

Strengths:
\- Strong empirical results. MCTTA achieves SOTA performance in Weakly Supervised Semantic Segmentation in Pascal VOC 2012 and COCO 2014 datasets.
\- Experiments and ablation studies are extensive.
\- Flexible design to work with multiple paradigms.

Weaknesses:
\- Lack of novelty. MCTTA seems like an incremental engineering improvement on the MCTformer+ architecture. CTP can be seen as a token initializer. Split Weighted Softmax is used to solve the imbalance of tokens. Hierarchical Fusion is borrowed from Fully Supervised Semantic Segmentation (as stated in the manuscript)
\- Misleading claim. The proposed method is called Adapter; however, it was designed as an extension for MCTformer+. If the proposed method is truly an adapter, the authors should have shown how the framework adapts to other methods.
\- Lack of supporting evidence. The manuscript states several weaknesses of the WSSS architecture, but lacks supporting evidence or is not backed by prior works.
\- Unclear distinction between single-stage and multi-stage paradigm. Although the manuscript states that using a pretrained model can be classified as single stage, the authors use a pretrained classifier to generate a pseudo label (or seed) and use that pseudo label to train the segmentation network while freezing the classification network (Figure 9.b). This is similar to multi-stage settings as defined in section IV.D.1. Other single-stage methods, such as [3][4] they tried to learn the segmentation map directly from the classification label.

Additional Comments and Questions:
\- The reviewer cannot find anything similar to "achieving rapid convergence in multi-class token prediction is critical" in the MCTformer+ paper[1]. Can authors help point it out? Since, without it, the motivation for CTP is not clear. One problem is that even though without CTP, the method converges more slowly, the loss at the end is still close. The reviewer thinks that the main concerns should be the performance gap, not the training loss gap. For example, [2] had shown that, in image classification, Adam usually converges faster but has worse generalization performance.

[1] MCTformer+: Multi-Class Token Transformer for Weakly Supervised Semantic Segmentation - Xu [et.al](http://et.al/) (TPAMI 2024)
[2] Adaptive Inertia: Disentangling the Effects of Adaptive Learning Rate and Momentum - Xie [et.al](http://et.al/) (ICML 2022 oral)
[3] Token Contrast for Weakly-Supervised Semantic Segmentation - Ru [et.al](http://et.al/) (CVPR 2023)
[4] Single-Stage Semantic Segmentation from Image Labels - Araslanov and Roth (CVPR 2020)

Additional Questions:
Is the work within the scope of the journal?: Clearly within scope

Is the manuscript technically correct?: Some minor concerns that should be easily corrected without altering the contribution or conclusions

Is the technical contribution novel?: Moderate novelty, with clear extensions of existing methods/concepts

Is the technical contribution significant?: Limited contribution, of limited interest to the community, and unlikely to have any impact

Is the length of the manuscript appropriate to the contribution?: Length is appropriate

Are the references appropriate, without any significant omissions?: Complete list of references without any significant omissions

Are there any references that do not appear to be relevant?: All references are directly relevant to the contribution of the manuscript

Is the manuscript properly structured and clearly written?: Moderate issues of exposition that may require some time to correct, but do not substantially affect the ability to evaluate the technical content

If you are suggesting additional references they must be entered in the text box provided.  All suggestions must include full bibliographic information plus a DOI.
: N/A