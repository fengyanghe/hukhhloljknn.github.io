---
layout: landing
title: Table of Contents
description: 冯旸赫的研究档案.
image: assets/images/head.jpg
nav-menu: yes
---

<!-- Banner -->
<!-- Note: The "styleN" class below should match that of the header element. -->
<section id="banner" class="style2">
	<div class="inner">
		<span class="image">
			<img src="{{ site.baseurl }}/{{ page.image }}" alt="" />
		</span>
		<header class="major">
			<h1>Table of Contents</h1>
		</header>
		<div class="content">
			<p>{{ page.description }}</p>
		</div>
	</div>
</section>

<!-- Main -->
<div id="main">
<section id="one" class="spotlights">
	{% for post in site.posts %}
		<section>
			<a href="{{ post.url }}" class="image">
				<img src="{{ post.image }}" alt="" data-position="top center" />
			</a>
			<div class="content">
				<div class="inner">
					<header class="major">
						<h3>{{ post.title }}</h3>
					</header>
					<p>{{ post.excerpt }}</p>
					<ul class="actions">
						<li><a href="{{ post.url }}" class="button">More</a></li>
					</ul>
				</div>
			</div>
		</section>
	{% endfor %}
</section>

</div>
