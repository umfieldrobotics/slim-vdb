---
# Front matter. This is where you specify a lot of page variables.
layout: default
title:  "Example Page"
date:   2023-06-16 03:03:01 -0400
description: >- # Supports markdown
  This is the main project page and the page used to demonstrate how this works with all of the options for the Front Matter present
show-description: true

# Add page-specific mathjax functionality. Manage global setting in _config.yml
mathjax: true
# Automatically add permalinks to all headings
# https://github.com/allejo/jekyll-anchor-headings
autoanchor: true

# Preview image for social media cards
image:
  path: https://cdn.pixabay.com/photo/2019/09/05/01/11/mountainous-landscape-4452844_1280.jpg
  height: 100
  width: 256
  alt: Random Landscape

# Only the first author is supported by twitter metadata
authors:
  - name: Someone
    footnotes: 1
    avatar: https://upload.wikimedia.org/wikipedia/commons/9/9f/Old_wikipedia_logo.png
  - name: Adam Li
    url: https://buildingatom.io
    email: person [at] example.edu
    mailto: person@example.edu
    footnotes: 2
    avatar: https://upload.wikimedia.org/wikipedia/commons/a/ac/Default_pfp.jpg
  - name: Person 3
    email: someemail@example.com
    footnotes: 3
  - name: Guy 4
    mailto: combined@example.com
    footnotes: 3
  - name: Another Person

# If you just want a general footnote, you can do that too.
# See the sel_map and armour-dev examples.
author-footnotes:
  1: >- # Supports markdown
    You can add random extra footnotes
  2: And include websites or emails which are detached from their mailto
  3: You can also just specify the email and not have a mailto, or if there's a mailto you want to use, you can specify only that

links:
  - icon: arxiv
    icon-library: simpleicons
    text: Arxiv HP
    url: https://arxiv.org/
  - icon: bi-map
    icon-library: bootstrap-icons
    text: SEL_Map Ex.
    url: /project-pages/sel_map
  - icon: bi-bezier2
    icon-library: bootstrap-icons
    text: ARMOUR Ex.
    url: /project-pages/armour-dev
  - icon: github
    icon-library: simpleicons
    text: Code
    url: https://github.com/BuildingAtom/project-pages
  - icon: https://upload.wikimedia.org/wikipedia/commons/c/c8/Black_W_for_promotion.png
    icon-library: local
    text: Wikipedia
    url: https://wikipedia.org/


# End Front Matter
---

{% include sections/authors %}
{% include sections/links %}

# Section Break
---

# As you can see, the authors and link are just includes

In other words, I can actually move them wherever I want, or place them here again.
The data is provided by the front matter.
Let's try flipping the order.

{% include sections/links %}
{% include sections/authors %}

---

There is quite a bit of flexibility to how you structure the text too.
Here, I'm going to make a justified grey content block with the heading outside.

# [Content](#content)
<div markdown="1" class="content-block grey justify no-pre">
some text

Try clicking this heading, this shows the manually defined header anchor, but if you do this, you should do it for all headings.
</div>

I made this look right by adding the `no-pre` class.
If you don't include `markdown="1"` it will fail to render any markdown inside.

You can also make fullwidth embeds (this doesn't actually link to any video)
<div class="fullwidth">
<video controls="" style="background-color:black;width:100%;height:auto;aspect-ratio:16/9;"></video>
</div>

<div markdown="1" class="content-block grey justify">
# Topic inside of the content block

Lorem ipsum dolor sit amet Consectetur adipiscing elit Integer molestie lorem at massa.

![Alt Text](https://cdn.pixabay.com/photo/2019/09/05/01/11/mountainous-landscape-4452844_1280.jpg "Random Image")
</div>

# Topic outside of content block

![Alt Text](https://cdn.pixabay.com/photo/2019/09/05/01/11/mountainous-landscape-4452844_1280.jpg "Random Image")

Lorem ipsum dolor sit amet Consectetur adipiscing elit Integer molestie lorem at massa.

## This is how we can get the image at 100%

<div markdown="1" class="fullwidth">
![Alt Text](https://cdn.pixabay.com/photo/2019/09/05/01/11/mountainous-landscape-4452844_1280.jpg "Random Image")
</div>

## And this is how we can get the image closer

<div markdown="1" class="no-pre">
![Alt Text](https://cdn.pixabay.com/photo/2019/09/05/01/11/mountainous-landscape-4452844_1280.jpg "Random Image")
</div>

Lorem ipsum dolor sit amet Consectetur adipiscing elit Integer molestie lorem at massa.

<div markdown="1" class="cabin">
It's also possible to specify a new font for a specific section
</div>

<div markdown="1" class="jp">
## See? 1
</div>

And you can also <span class="cabin">change it in the middle</span>, though that's a bit more problematic for other reasons.

To specify fonts, just use Google Fonts and update `_data/fonts.yml`.
Any fonts you add as extra fonts at the bottom become usable fonts in the body of the post.

There are also tools to grab icons from other repos.
Just use the following:
{% include util/icons icon='github' icon-library='simpleicons' -%}
, and you'll be able to add icons from any library you have enabled that is supported.

This uses the liquid template engine for importing.
If you include the - at the start of end of such a line, it say to discard all whitespace before or after.
In order to keep the comma there, we added the -.
This is what happens:
{% include util/icons icon='github' icon-library='simpleicons' %}
, when you don't have it (notice the space).

And if you have mathjax enabled in `_config.yml` or in the Front Matter as it is here, you can even add latex:

$$
\begin{align*}
  & \phi(x,y) = \phi \left(\sum_{i=1}^n x_ie_i, \sum_{j=1}^n y_je_j \right)
  = \sum_{i=1}^n \sum_{j=1}^n x_i y_j \phi(e_i, e_j) = \\
  & (x_1, \ldots, x_n) \left( \begin{array}{ccc}
      \phi(e_1, e_1) & \cdots & \phi(e_1, e_n) \\
      \vdots & \ddots & \vdots \\
      \phi(e_n, e_1) & \cdots & \phi(e_n, e_n)
    \end{array} \right)
  \left( \begin{array}{c}
      y_1 \\
      \vdots \\
      y_n
    \end{array} \right)
\end{align*}
$$

You can also treat a section of text as a block, and use kramdown's block attribution methods to change fonts.
You can see at the end of this section in the markdown that I do just that
{: class="cabin"}

# Videos
{% include util/video
content="http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4"
autoplay-in-frame=false
autoplay-on-load=false
preload="auto"
hide_controls=false
picture_in_picture=true
muted=true
loop=false
playsinline=true
pause_on_click=true
poster="https://upload.wikimedia.org/wikipedia/commons/9/90/Elephants_Dream_s1_proog.jpg"
width="90%"
style="display:block; margin:auto;"
%}

# "Gridding"
<div class="multicontent-container">
<div class="multicontent-item">
{% include util/video
content="http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4"
autoplay-in-frame=true
muted=true
poster="https://upload.wikimedia.org/wikipedia/commons/9/90/Elephants_Dream_s1_proog.jpg"
 %}
autoplayed when ~25% in view
</div>
<div class="multicontent-item">
{% include util/video
content="http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4"
autoplay-on-load=true
muted=true
poster="https://upload.wikimedia.org/wikipedia/commons/9/90/Elephants_Dream_s1_proog.jpg"
 %}
autoplayed when video loaded
</div>
<div class="multicontent-item">
{% include util/video
content="http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4"
poster="https://upload.wikimedia.org/wikipedia/commons/9/90/Elephants_Dream_s1_proog.jpg"
 %}
</div>
</div>

# "Gridding" Tighter
<div class="multicontent-container tighter">
<div class="multicontent-item" markdown="1">
![Alt Text](https://cdn.pixabay.com/photo/2019/09/05/01/11/mountainous-landscape-4452844_1280.jpg "Random Image")
Random text to talk about in this one grid box
</div>
<div class="multicontent-item" markdown="1">
![Alt Text](https://cdn.pixabay.com/photo/2019/09/05/01/11/mountainous-landscape-4452844_1280.jpg "Random Image")
</div>
<div class="multicontent-item" markdown="1">
![Alt Text](https://cdn.pixabay.com/photo/2019/09/05/01/11/mountainous-landscape-4452844_1280.jpg "Random Image")
</div>
<div class="multicontent-item" markdown="1">
![Alt Text](https://cdn.pixabay.com/photo/2019/09/05/01/11/mountainous-landscape-4452844_1280.jpg "Random Image")
</div>
<div class="multicontent-item" markdown="1">
![Alt Text](https://cdn.pixabay.com/photo/2019/09/05/01/11/mountainous-landscape-4452844_1280.jpg "Random Image")
</div>
</div>

<div markdown="1" class="content-block grey justify">
# This is a really long heading block so I can see if justify breaks the heading, and make sure that headings don't get justify unless they are explicitly classed with justify like the following heading

# This is the following really long heading block so I can see if justify breaks the heading, and make sure that only this heading is justified because it has the explicit tag
{: class="justify"}
</div>

<div markdown="1" class="content-block grey justify">
# Citation

*Insert whatever message*

```bibtex
@article{nash51,
  author  = "Nash, John",
  title   = "Non-cooperative Games",
  journal = "Annals of Mathematics",
  year    = 1951,
  volume  = "54",
  number  = "2",
  pages   = "286--295"
}
```
</div>
