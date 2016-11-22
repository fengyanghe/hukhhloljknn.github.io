---
layout: post
title: Generic
description: Lorem ipsum dolor est
image: assets/images/myself.jpg
nav-menu: yes
---

<ul class="links">
	{% for page in site.pages %}
	    {% if page.layout == "home" %}
	        <li><a href="{{ page.url | prepend: site.baseurl | remove: "index.html" }}">{{ page.title }}</a></li>
		{% endif %}
	{% endfor %}
	{% for page in site.posts %}
	    {% if page.layout != "home" %}
	        <li><a href="{{ page.url | prepend: site.baseurl }}">{{ page.title }}</a></li>
	    {% endif %}
	{% endfor %}