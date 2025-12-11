---
# Front matter. This is where you specify a lot of page variables.
layout: default
title:  "SLIM-VDB"
date:   2025-12-10 10:00:00 -0500
description: >- # Supports markdown
  A Real-Time 3D Probabilistic Semantic Mapping Framework
show-description: true

# Add page-specific mathjax functionality. Manage global setting in _config.yml
mathjax: true
# Automatically add permalinks to all headings
# https://github.com/allejo/jekyll-anchor-headings
autoanchor: false

# Preview image for social media cards
image:
  path: /assets/mainfig_compressed.jpg
  height: 490
  width: 800
  alt: SLIM-VDB Main Figure

# Only the first author is supported by twitter metadata
authors:
  - name: Anja Sheppard
    email: anjashep@umich.edu
  - name: Parker Ewen
  - name: Joey Wilson
  - name: Advaith V. Sethuraman
  - name: Benard Adewole
  - name: Anran Li
  - name: Yuzhen Chen
  - name: Ram Vasudevan
  - name: Katherine A. Skinner

# If you just want a general footnote, you can do that too.
# See the sel_map and armour-dev examples.
author-footnotes: |
  <br> All authors affiliated with the department of Department of Robotics of the University of Michigan, Ann Arbor.

links:
  - icon: arxiv
    icon-library: simpleicons
    text: ArXiv
    url: https://arxiv.org/
  - icon: github
    icon-library: simpleicons
    text: Code
    url: https://github.com/umfieldrobotics/slim-vdb

# End Front Matter
---

{% include sections/authors %}
{% include sections/links %}

---

# [Overview Videos](#overview-videos)

<!-- BEGIN OVERVIEW VIDEOS -->
<div class="fullwidth">
  {% include util/video
    content="assets/mainvid_compressed.mp4"
    poster="assets/thumb/mainvid_compressed.png"
    autoplay-on-load=true
    preload="none"
    muted=true
    loop=true
    playsinline=true
    %}
  <p style="text-align:center; font-weight:bold;">SLIM-VDB Overview</p>
</div><!-- END OVERVIEW VIDEOS -->

<!-- BEGIN ABSTRACT -->
<div markdown="1" class="content-block justify grey">

# [Abstract](#abstract)

This paper introduces SLIM-VDB, a new lightweight semantic mapping system with probabilistic semantic fusion for closed-set or open-set dictionaries.
Advances in data structures from the computer graphics community, such as OpenVDB, have demonstrated significantly improved computational and memory efficiency in volumetric scene representation. 
Although OpenVDB has been used for geometric mapping in robotics applications, semantic mapping for scene understanding with OpenVDB remains unexplored.
In addition, existing semantic mapping systems lack support for integrating both fixed-category and open-language label predictions within a single framework.
In this paper, we propose a novel 3D semantic mapping system that leverages the OpenVDB data structure and integrates a unified Bayesian update framework for both closed- and open-set semantic fusion. 
Our proposed framework, SLIM-VDB, achieves significant reduction in both memory and integration times compared to current state-of-the-art semantic mapping approaches, while maintaining comparable mapping accuracy.
An open-source C++ codebase with a Python interface is available at https://github.com/umfieldrobotics/slim-vdb.

</div> <!-- END ABSTRACT -->

<!-- BEGIN APPROACH -->
<div markdown="1" class="justify">

# [Approach](#approach)

![method_overview](./assets/diagram.png)
{: class="fullwidth no-pre"}

<!-- # Contributions -->
SLIM-VDB takes advantage of the highly optimized 3D volumetric representation presented in OpenVDB, a library used in the computer graphics community. Our work integrates semantic fusion at a voxel level with OpenVDB, resulting in a lightweight mapping library that can handle both open-set and closed-set semantics. The key component is Bayesian semantic fusion: this work takes advantage of Dirichlet-Categorical conjugacy and Normal-Normal Inverse Gamma conjugacy to tractably handle different semantic predictions from a network.

Our key contributions are:
1. A novel framework that builds on OpenVDB to enable lightweight, memory-efficient semantic mapping.
2. A unified Bayesian inference framework that enables either closed-set or open-set semantic estimation.
3. An open-source C++ library with a Python interface for easy integration with robotics applications.

</div><!-- END METHOD -->

<!-- START CITATION -->
<div markdown="1" class="content-block grey justify">
 
# [Citation](#citation)

This project was developed in the [Field Robotics Group](https://fieldrobotics.engin.umich.edu/) at the University of Michigan - Ann Arbor.

```bibtex
@article{sheppard2025slimvdb,
  author         = {Sheppard, Anja and Ewen, Parker and Wilson, Joey and Sethuraman, Advaith V. and Adewole, Benard and Li, Anran and Chen, Yuzhen and Vasudevan, Ram and Skinner, Katherine A.},
  title          = {SLIM-VDB: A Real-Time 3D Probabilistic Semantic Mapping Framework},
  journal        = {Robotics Automation and Letters},
  volume         = {TBD},
  year           = {2025},
  number         = {TBD},
  article-number = {TBD},
  url            = {TBD},
  doi            = {TBD}
}
```
</div>
<!-- END CITATION -->

<!-- below are some special scripts -->
<script>
window.addEventListener("load", function() {
  // Get all video elements and auto pause/play them depending on how in frame or not they are
  let videos = document.querySelectorAll('.autoplay-in-frame');

  // Create an IntersectionObserver instance for each video
  videos.forEach(video => {
    const observer = new IntersectionObserver(entries => {
      const isVisible = entries[0].isIntersecting;
      if (isVisible && video.paused) {
        video.play();
      } else if (!isVisible && !video.paused) {
        video.pause();
      }
    }, { threshold: 0.25 });

    observer.observe(video);
  });

  // document.addEventListener("DOMContentLoaded", function() {
  videos = document.querySelectorAll('.autoplay-on-load');

  videos.forEach(video => {
    video.play();
  });
});
</script>