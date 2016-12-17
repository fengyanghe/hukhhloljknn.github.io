# modified on Forty - Jekyll Theme

I use the template named A Jekyll version of the "Forty" theme by [HTML5 UP](https://html5up.net/).

	- create a new post command: rake post title="new post"

	- create a site command: jekyll serve

Equation edit

	Shown by `<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>`
	
>inline style:
> 
>      `$...$`
>>or
> 
>      `\(....\)`
	
> otherwise:
> 
>      $$
>      ...
>      $$
>>or
> 
>      `\[...\]`

Notes:

1. Menu defined in header.html load all pages titles of which layout is landing type. There are only two pages which layout is landing type, one is landing.md. It filters all of my study paper which the category is "paper". The other is landingWeekly.md. It filters all of awesome papers which the category is "weekly". Therefore, one can create a new md of which the layout is landing if one want to add the new menu item.

2. The landing.md or landingWeekly.md is the summary for all of post pages. The introduce is {{ post.excerpt }} which refers the content end up ```<!-- excerpt -->```.

3. The layout named post can show the image at the first of the content, in contrast, the page layout don't solve image.  
