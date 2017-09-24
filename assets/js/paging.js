/* the paing.js is finished by LiaoJiang 2017/9/23
*  for Jekyll in GitHub Pages the blog:
* fengyanghe.github.io
*/
var urlArray = [];
var picArray = [];
var titleArray = [];
//var abstract = []; //摘要
var limit = 0;    //每页项目数
var pagecount = 0; //the pages count
var crtpage = 1;  //the current page index start from 1
var itemscount = 0;  //总文章数
var splitstr = "#-#"; //分隔符
var itemIdstr = "one"; //每篇文章的展示块 ID前缀
var idArray = ["url", "title", "image", "url_"];
var sIndex = 0;    //显示项目开始序号
var showcount = 0; //本页显示项目数量
var abstNodeList = []; //摘要节点列表

//获取所有子节点列表,排除孙节点
function getChildNodes(ele){
   var childArr=ele.children,
         childArrTem=new Array();  //  临时数组，用来存储符合条件的节点
    for(var i=0,len=childArr.length;i<len;i++){
        if(childArr[i].nodeType==1){
            childArrTem.push(childArr[i]);  // push() 方法将节点添加到数组尾部
        }
    }
    return childArrTem;
}

//初始化摘要节点列表
function init_abstract() {
    //var abscount = abstract.length;
   // abstract[abscount] = abstractstr;
    abstNodeList = getChildNodes(document.getElementById("ablist"));
}

//init pages, limit...
function init_para(page_postcount, urllist, piclist, titlelist) {
    limit = page_postcount;

    urlArray = urllist.split(splitstr);
    urlArray.pop();

    picArray = piclist.split(splitstr);
    picArray.pop();

    titleArray = titlelist.split(splitstr);
    titleArray.pop();

    itemscount = urlArray.length;
    if (itemscount % limit == 0)
     {
        pagecount = itemscount / limit;
     } 
     else 
     {
        pagecount = parseInt(itemscount / limit) + 1;
     }
}

//添加摘要节点
function set_abstract() {
    for (var i = 0; i < showcount; i++) {
        var abstindex = sIndex + i;
        var idstr = "abst" + (i+1);
        var clonedNode = abstNodeList[abstindex].cloneNode(true);
        clonedNode.setAttribute("id", "div-" + crtpage + "-" + i); // 修改一下id 值，避免id 重复,并方便切换页面后删除节点
        clonedNode.setAttribute("style", "display: block;");
        document.getElementById(idstr).appendChild(clonedNode);
    }
}

//切换页面后删除节点
function  removeAbstNodes() {
    for (var i = 0; i < showcount; i++) {
        var idstr = "abst" + (i+1);               //容器Id
        var abstId = "div-" + crtpage + "-" + i;  //摘要节点Id
        var abstNode = document.getElementById(abstId); 
        document.getElementById(idstr).removeChild(abstNode); 
    }
}

//do the selection
/*optnum:
*-1:previous page
*-2:next page
*other:select page num
*/
function sep_func(optnum) {
    var idstr = "";    //页面ID标识变量

     removeAbstNodes(); //在原有页面的项目数，页面序号等信息被覆盖前，删除原页面的摘要节点

    switch(optnum){
        case(-1):
            crtpage --;
            break;
        case(-2):
            crtpage ++;
        default:
            crtpage = optnum;
        break;
    }


    sIndex = (crtpage - 1) * limit;
    showcount = itemscount - sIndex;
    if (showcount > limit) 
    {
        showcount = limit;
    } 

    //赋值
    for (var i = 0; i < showcount; i++) {
        //url值更新
        idstr = idArray[0] + (i + 1);
        document.getElementById(idstr).href = urlArray[sIndex + i];

        //title值更新
        idstr = idArray[1] + (i + 1);
        document.getElementById(idstr).innerHTML = titleArray[sIndex + i];

        //pic值更新
        idstr = idArray[2] + (i + 1);
        document.getElementById(idstr).src = picArray[sIndex + i];

         //more url值更新
        idstr = idArray[3] + (i + 1);
        document.getElementById(idstr).href = urlArray[sIndex + i];

        //摘要更新
        //idstr = idArray[4] + (i + 1);
       // document.getElementById(idstr).innerHTML = sIndex + i;
    }
    set_abstract();

    //显示showcount个项目
    for (var i = 0; i < showcount; i++) {
        idstr = itemIdstr + (i + 1);
        if (document.getElementById(idstr).style.display == "none") 
        {
            document.getElementById(idstr).style.display = "";//block会影响CSS布局
        } 
    }

    //隐藏应该缺失的默认项目
    for (var i = 0; i < limit - showcount; i++) {
        idstr = itemIdstr + (i + showcount + 1);
        document.getElementById(idstr).style.display = "none"; 
    }

    //获取分页的文本样式
    var pagehtml = "<div id='paging' style='text-align:center'>";
    var firsthtml = "", prevhtml = "", choosehtml = "", nexthtml = "", lasthtml = "";
    var endhtml = "</div>";

    //first page and privious page html
    if (crtpage == 1) 
    {
        firsthtml = "<em>&lt;&lt;</em>";
        prevhtml = "<em>prev</em>"
    } 
    else 
    {
        firsthtml = "<a href=\"javascript:sep_func(1)\" title='first page'>&lt;&lt;</a>";
        prevhtml = "<a href=\"javascript:sep_func(" + (crtpage - 1) + ")\">prev</a>";
    }

    //the number html
    if(pagecount<=10){        //总页数小于等于十页 
        for (count=1;count<=pagecount;count++) 
        {    if(count!=crtpage) 
            { 
                choosehtml = choosehtml + "<a href='javascript:void(0)' onclick='sep_func("+count+")'>"+count+"</a>"; 
            }else{ 
                choosehtml = choosehtml + "<em>"+count+"</em>"; 
            } 
            if (count < pagecount)
            {
                choosehtml += " ";
            } 
        } 
    } 
    if(pagecount>10){  //总页数大于十页 
        if(parseInt((crtpage-1)/10) == 0)  //在第一组
        {             
            for (count=1;count<=10;count++) 
            {    if(count!=crtpage) 
                { 
                    choosehtml = choosehtml + "<a href='javascript:void(0)' onclick='sep_func("+count+")'>"+count+"</a>"; 
                }else{ 
                    choosehtml = choosehtml + "<em>"+count+"</em>"; 
                } 
                if (count < 10)
                {
                    choosehtml += " ";
                } 
            } 
            choosehtml = choosehtml + "<a href='javascript:void(0)' onclick='sep_func("+count+")' title='next page group'> ... </a>"; 
        } 
        else if(parseInt((crtpage-1)/10) == parseInt(pagecount/10))  //最后一组
        {     
            choosehtml = choosehtml + "<a href='javascript:void(0)' onclick='sep_func("+(parseInt((crtpage-1)/10)*10)+")' title='previous page group'>...</a>"; 
            for (count=parseInt(pagecount/10)*10+1;count<=pagecount;count++) 
            {    if(count!=crtpage) 
                { 
                    choosehtml = choosehtml + "<a href='javascript:void(0)' onclick='sep_func("+count+")'>"+count+"</a>"; 
                }else{ 
                    choosehtml = choosehtml + "<em>"+count+"</em>"; 
                } 
                if (count < pagecount)
                {
                    choosehtml += " ";
                } 
            } 
        } 
        else 
        {   
            choosehtml = choosehtml + "<a href='javascript:void(0)' onclick='sep_func("+(parseInt((crtpage-1)/10)*10)+")' title='previous page group'>...</a>"; 
            for (count=parseInt((crtpage-1)/10)*10+1;count<=parseInt((crtpage-1)/10)*10+10;count++) 
            {         
                if(count!=crtpage) 
                { 
                    choosehtml = choosehtml + "<a href='javascript:void(0)' onclick='sep_func("+count+")'>"+count+"</a>"; 
                }else{ 
                    choosehtml = choosehtml + "<em>"+count+"</em>"; 
                } 
            } 
            choosehtml = choosehtml + "<a href='javascript:void(0)' onclick='sep_func("+count+")' title='next page group'>...</a>"; 
        } 
    }     

    //the next page html and the last page select
    if (crtpage >= pagecount)  //if the pagecount is 0, no link
    {
        nexthtml = "<em>next</em>";
        lasthtml = "<em>&gt;&gt;</em>"
    } 
    else 
    {
        nexthtml = "<a href=\"javascript:sep_func("+(crtpage + 1)+")\">next</a>";
        lasthtml = "<a href=\"javascript:sep_func(" + pagecount + ")\" title='last page'>&gt;&gt;</a>";
    }

    pagehtml += firsthtml + " " + prevhtml + " " + choosehtml + " " + nexthtml + " " + lasthtml + endhtml;
    document.getElementById("paging").innerHTML = pagehtml;				//set the paging div content
    document.body.scrollTop = document.documentElement.scrollTop = 100; //go to the top
}