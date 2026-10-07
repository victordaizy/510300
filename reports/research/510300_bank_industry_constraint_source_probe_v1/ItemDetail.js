; (function (app) {
  app.filter("ljkh", [function ($sce) {  //红头栏目详情 []改为〔〕
    
    return function (originalStr) {
      if(originalStr){
        var tempTxt = originalStr.replace("[","〔");
        return tempTxt.replace("]","〕");
      }
    }
        
  }]);
  var branchArr = ["北京", "天津", "河北", "山西", "内蒙古", "辽宁", "吉林", "黑龙江", "上海", "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南", "湖北", "湖南", "广东", "广西", "海南", "重庆", "四川", "贵州", "云南", "西藏", "陕西", "甘肃", "青海", "宁夏", "新疆", "大连", "宁波", "厦门", "青岛", "深圳"]
  app.filter("branchFilter", [function ($sce) {  //红头栏目详情 []改为〔〕
    
    return function (originalStr) {
      if(originalStr){
       for(var i=0;i<branchArr.length;i++) {
         if(originalStr === branchArr[i]) {
           return originalStr + '监管局'
         }
       }
      }
      return originalStr
    }
        
  }]);
  //   中国银行业监督管理委员会(0-12)    中国保险监督管理委员会(13-27)   中国银行保险监督管理委员会(28->)    
  var documentNoList = ['中国银行业监督管理委员会令','中国银监会令','银监通','银监函','银监发','银监复','银监会公告','银监办发','银监办通','银监办函','银监办便函','融资担保发','融资担保函','中国保险监督管理委员会令','保监发','保监厅发','保监厅函','保监发改','保监稽查','保监人身险','保监寿险','保监财会','保监产险','保监统信','保监消保','保监中介','保监资金','资金部函','中国银行保险监督管理委员会令','银保监规','银保监发','银保监复','银保监函','银保监会公告','银保监办发','银保监撤销许可','银保监注销许可'];
  var documentNoTitle = '';
  app.filter("documentNoTxt", [function ($sce) {  // 红头标题按照文号区分
    
    return function (originalStr,datafrom) {
      if(!originalStr || originalStr === '其他' || originalStr === '无'){
        return documentNoTitle
      }else {
        tmpTxt = originalStr.split('[')[0];
        for (var i=0;i<documentNoList.length;i++){
          if(tmpTxt == documentNoList[i]){
            return documentNoTitle =  i <=12 ? '中国银行业监督管理委员会' : i <= 27 ? '中国保险监督管理委员会' : '中国银行保险监督管理委员会'
          }else {
            documentNoTitle = datafrom != null ? '' : '中国银行保险监督管理委员会';
          }
        }
        return documentNoTitle
      }
    }
        
  }]);
  function isGuiZhang(arr) {
    if (!arr || !(arr instanceof Array)) {
        return 1;
    }
    for (var  i = 0; i < arr.length; i++) {
      // 发布的栏目是否包含规章栏目 
      if(arr[i].ItemID == 4214){
        return 2;
      }
    }
    return 1;
  }
  function fixGeneraltype(arr) {
    var itemArr = ['4222','860'];
    for (var  i = 0; i < arr.length; i++) {
      for (var  ii = 0; ii < itemArr.length; ii++) {
        if(arr[i].ItemID == itemArr[ii]){
          return '0';
        }
      }
    }
    return '1';
  }
    app.controller('itemDetailCtrl', function ($scope, global, $rootScope,$timeout) {
        // console.log($rootScope.image_path, 212121)
        // add scope.images
          //判断是否为静态化页面
        //   $(document).ready(function () {
        //     if (window.location.href.indexOf('static=1') > -1) {
        //         //console.log('static');
        //         $.getScript('/cn/js/common/ngclean.js', function () {
        //             $.ngClean($("body"), { removeAttrs: ["ng-app", "ng-controller", "ng-repeat", "ng-init", "ng-model"], removeClasses: ["ng-binding", "ng-scope"] });
        //             //console.log(staticHtml);
        //             //$(".main").html(staticHtml);
        //         });

        //     }
        // });
       
        // add scope.images
        $scope.isIE8 = false;
        var version = 8.0;  
        var ua = navigator.userAgent.toLowerCase();  
        var isIE = ua.indexOf("msie")>-1;  
        var safariVersion;  
        if(isIE){  
        safariVersion =  ua.match(/msie ([\d.]+)/)[1];  
        }  
        if(safariVersion <= version ){  
          $scope.isIE8 = true;
        }
        var docId = getParam("docId");
        var zfxx_itemId = getParam("itemId");
        var itemType = getParam("type");
        var newbuilddate = "";
        var newpublishDate = "";
        var dadeline = new Date('2019-12-30').getTime();
        var itemId = getParam("itemId");

        $scope.generaltype = getParam("generaltype");
        // if(itemId && fixGeneraltype(itemId)) { //generaltype == 1不需要展示红头的处理
        //   $scope.generaltype = '0';
        // }
        if ($scope.generaltype == '1') {
            $scope.showSource = false;//政府信息公开页面中generaltype==1为红头的文件 不显示来源
        } else {
            $scope.showSource = true;//政府信息公开页面中generaltype!=1基本上不是红头 显示来源但此栏目下的法律行政法规栏目除外，具体在面包屑中处理了
        }

        if (docId == undefined || docId == '') {
            return;
        }
        //  文章下载
        $scope.downloadPage = function () {
            if (!window.location.origin) {
                windowURL = window.location.protocol + "//" + window.location.hostname + (window.location.port ? ':' + window.location.port : '');
            } else {
                windowURL = window.location.origin;
            }
            // window.open(windowURL+"/cbircweb/download/downloadPdf?docId=" + docId );
            window.open(windowURL + "/cbircweb/download/downloadPdf?docId=" + docId + "&generaltype=" + $scope.generaltype + "&itemId=" + itemId);


        }
        $scope.docId = docId;
        $scope.rulesDocFileDownload = function (type) {
          // if(itemId==927){
          //     var zcfg=0;
          // }else{
          //     var zcfg=1;
          // }
          var file_urlnew = type ==1? "/cbircweb/download/downloadguizDoc" : "/cbircweb/download/downloadguizPdf";
          // var file_urlnew2=windowURL + "/cbircweb/download/downloadPdf?docId=" + docId+"&zcfg=1";
          $.ajax({
              url: file_urlnew,
              type:"GET",
              data:{docId:docId },
              error: function (xhr, error, ex) {
                  if (xhr.status == '200') {
                      window.location.href = file_urlnew;
                  } else if (xhr.status == '404') {
                      alert("文件不存在！")
                  }else if (xhr.status == '423') {
                      alert("文件尚未生成！")
                  }
              },
              success: function () {
                  window.location.href = file_urlnew+ "?docId=" + docId;
              }
          });
       }
        global.getCDN({ url: '/DocInfo/SelectByDocId', params: { docId: docId } }, function (res) {
            if (res.rptCode == 200 && res.data != null) {
                $scope.isGuiZhang = isGuiZhang(res.data.listTwoItem);
                if(window.location.href.indexOf('governmentDetail') != -1 && fixGeneraltype(res.data.listTwoItem) == '0') {
                    $scope.generaltype = '0';
                    $scope.hideSource = true;
                }
                newbuilddate = new Date(res.data.builddate).getTime();
                newpublishDate = new Date(res.data.publishDate).getTime();
                $scope.data = res.data;
                if(res.data.aviCodeUrl && $scope.isIE8){
                  $("#embed").html('');
                  var vhtml = '<div class="ie-off" style="width:740px;height:416px;background: #000;margin:0 auto;position:relative"><span style="color:#fff">本浏览器不支持视频播放，请您选择其他浏览器。</span></div>'
                  $("#embed").html(vhtml);
                  var videoHtml = '<span style="color:#fff;position:absolute;left:215px">本浏览器不支持视频播放，请您选择其他浏览器。</span><embed src="'+res.data.aviCodeUrl+'" allowScriptAccess="always" width="740" height="416" autostart="false"></embed>'
                  $(".ie-off").html(videoHtml);
                }
                
                //确认专题专栏下的文章怎么展示视频（一文多发？）
                //接受是否有视频的字段即可
                $timeout(function () {
                  var pArr = $('p>span');
                  for(var i=0;i<pArr.length;i++) {
                   if($(pArr[i]).length == 1 && ($(pArr[i])[0].innerHTML == '<br>' || $(pArr[i])[0].innerHTML == '<br />' ||$(pArr[i])[0].innerHTML == '<br/>' ||$(pArr[i])[0].innerHTML == '<BR>' )){
                     $(pArr[i]).addClass('zh-child')
                   }
                  }
                },60);
                // $rootScope.image_path = "images";
                $rootScope.image_path = "images";
                if (global.isRequestSkinJson) {
                    $.ajax({
                        url: '/cn/css/common/data_.json',
                        type: 'get',
                        dataType: 'json',
                        success: function (data) {
                            if (data.data == 'grayscale') {
                                $("body").addClass("grayscale");
                                $rootScope.image_path = "images_gray";
                                $rootScope.isGray = true;
                                $rootScope.svgGrayscaleFilter = "url('#grayscale')";
                                var ieret = global.getIEVersion();
        
                                if (ieret != -1) {
                                    if (ieret >= 10 || ieret == 'edge') {
                                        setTimeout(function(){
                                            if($('#wenzhang-content img').length!=0){
                                                grayscale($('#wenzhang-content img'),function(){
                                                    $rootScope.showSVG = true;
                                                });
                                            }
                                           
                                            setTimeout(function(){
                                                $scope.$apply();
                                            },900)
                                            $scope.$apply();
                                        },1000)
                                    } else {
                                        $rootScope.showSVG = false;
                                    }
                                }
                            } else {
                                $rootScope.image_path = "images";
                                $rootScope.isGray = false;
                                $rootScope.svgGrayscaleFilter = "";
                            }
                        },
                        error: function () {
                            global.getCDN({ url: '/Skin/getCurkind' }, function (res) {
                                if (res.rptCode == 200) {
                                    if (res.data == 'grayscale') {
                                        $("body").addClass("grayscale");
                                      
                                        $rootScope.image_path = "images_gray";
                                        $rootScope.isGray = true;
                                        $rootScope.svgGrayscaleFilter = "url('#grayscale')";
                                        var ieret = global.getIEVersion();
        
                                        if (ieret != -1) {
                                            if (ieret >= 10 || ieret == 'edge') {
                                                // $rootScope.showSVG = true;
                                                // grayscale($('#wenzhang-content'));
                                                setTimeout(function(){
                                                    if($('#wenzhang-content img').length!=0){
                                                        grayscale($('#wenzhang-content img'),function(){
                                                            $rootScope.showSVG = true;
                                                        });
                                                    }
                                                   
                                                    setTimeout(function(){
                                                        $scope.$apply();
                                                    },900)
                                                    $scope.$apply();
                                                },1000)
                                              
                                            } else {
                                                $rootScope.showSVG = false;
                                            }
                                        }
                                    } else {
                                        $rootScope.image_path = "images";
                                        $rootScope.isGray = false;
                                        $rootScope.svgGrayscaleFilter = "";
                                    }
                                } else {
                                    $rootScope.image_path = "images";
                                    $rootScope.isGray = false;
                                    $rootScope.svgGrayscaleFilter = "";
                                }
                            })
                        }
                    })
                } else {
                    global.getCDN({ url: '/Skin/getCurkind' }, function (res) {
                        if (res.rptCode == 200) {
                            if (res.data == 'grayscale') {
                                $("body").addClass("grayscale");
                                $rootScope.image_path = "images_gray";
                                $rootScope.isGray = true;
                                $rootScope.svgGrayscaleFilter = "url('#grayscale')";
                                var ieret = global.getIEVersion();
        
                                if (ieret != -1) {
                                    if (ieret >= 10 || ieret == 'edge') {
                                       setTimeout(function(){
                                            if($('#wenzhang-content img').length!=0){
                                                grayscale($('#wenzhang-content img'),function(){
                                                    $rootScope.showSVG = true;
                                                });
                                            }
                                           
                                            setTimeout(function(){
                                                $scope.$apply();
                                            },900)
                                            $scope.$apply();
                                        },1000)
                                       
                                    } else {
                                        $rootScope.showSVG = false;
                                    }
                                }
                            } else {
                                $rootScope.image_path = "images";
                                $rootScope.isGray = false;
                                $rootScope.svgGrayscaleFilter = "";
                            }
                        } else {
                            $rootScope.image_path = "images";
                            $rootScope.isGray = false;
                            $rootScope.svgGrayscaleFilter = "";
                        }
                    })
                }
                // 设置多链接
                if (res.data.remark2) {
                    $scope.valueArr = res.data.remark2.match(/\[[^\]]+\]\s*[(|（][^)^）^(^（)]+[)|）]\s*/g)// 附录
                    var regExp2 = /\[[^\]]+\]\s*[(|（][^)^）^(^（)]+[)|）]\s*/g;

                    if (regExp2.test(res.data.remark2) == false) {

                        res.data.rules = "1";
                    } else {
                        res.data.rules = "0";

                        $scope.editSubContent = []
                        for (var i = 0; i < $scope.valueArr.length; i++) {
                            $scope.valueArr[i] = $scope.valueArr[i].replace(/(^\s*)|(\s*$)/g);
                            //解析存入的[标题](地址)

                            $scope.valueArr[i].replace(/\[(.*)\]\s*[\(|（](.*)[\)|）]/, function () {
                                $scope.editSubContent.push({ title: arguments[1], href: arguments[2] })
                                // console.log(arguments)
                            })
                        }

                    }

                } else {

                }
                // 设置多链接
                //    if (res.data.docUuid == "" || res.data.docUuid == null) {
                //     $scope.showTitle = true;
                //   } else if (itemType == "4") {
                //     $scope.showTitle = false;
                //   } else {
                //     $scope.showTitle = false;
                //   }    
                var itemId=getParam('itemId');
                if (res.data.docUuid == "" || res.data.docUuid == null) {
                    // $scope.showTitle = true;
                    // 文号显示
                        //标题显示情况：非年报情况下docUuid为null不论文号有否前端都手动拼上标题
                        $scope.showTitle = true;
                        if ((res.data.documentNo != "" && res.data.documentNo != null) && itemId == 928 ) {
                         //文号显示情况：非年报情况下docUuid为null同时文号不为空时手动加上文号
                         if((res.data.documentNo.indexOf("银保监会令") == -1)){
                            $scope.showDocNo = true;
                            itemId='';
                         }else{
                            $scope.showDocNo = false;
                         }
                         
                        }else{
                            $scope.showDocNo = false;
                        }
                    // 文号显示
                } else if (itemType == "4") {
                    $scope.showTitle = false
                } else {
                    $scope.showTitle = false;
                }

                // 从接口获取挂牌时间
                var sysBuildDate = '';
                global.getCDN({ url: '/Skin/getSysParams' }, function (re) {
                    if (re.rptCode == 200) {
                        sysBuildDate = re.data.ceremony_time;
                    }
                    // 如果发文日期不早于挂牌时间 红头展示“国家金融监督管理总局” 其他逻辑不变
                    if(new Date(res.data.builddate) >= new Date(sysBuildDate)) {
                        $scope.afterCeremony = true;
                        $scope.itemTitle = '国家金融监督管理总局';
                    } else {
                        $scope.afterCeremony = false;
                        $scope.itemTitle = '中国银行保险监督管理委员会';
                        // 判断若发布日期小于12月30，让发布日期取发文日期
                        //  1.发布日期大于12月30号，发文日期小于12月30号  处理： 发文日期以发布日期为准
                        //  2.发布日期小于12月30号，发文日期小于12月30号   处理：发布按发文日期为准
                        //  3.发布日期大于12月30号，发文日期大于12月30号   不处理（各自为准）
                        //  4.发布日期小于12月30号，发文日期大于12月30号   发布按发文
                        if (res.data.publishDate < '2019-12-30' && newbuilddate != 0 && res.data.builddate < '2019-12-30') {
                            var pubstr = res.data.publishDate.substr(0, 10);
                            res.data.publishDate = res.data.publishDate.replace(pubstr, res.data.builddate);
        
                        }
                        //  console.log('发布日期大于12月30、发文日期小于12月30号');
                        //      } 
                        if (res.data.publishDate < '2019-12-30' && newbuilddate != 0 && res.data.builddate > '2019-12-30') {
                            var pubstr2 = res.data.publishDate.substr(0, 10);
                            res.data.publishDate = res.data.publishDate.replace(pubstr2, res.data.builddate);
        
                        }
                        //判断红头文件银保监会、银监会、保监会
                        if (res.data.publishDate >= '2018-03-28' && res.data.datafrom == 0 || res.data.datafrom == null || res.data.datafrom == 2) {
                            $scope.isYinbaojian = true;  //中国银行保险监督管理委员会
                        } else if (res.data.datafrom == 0 && res.data.publishDate < '2018-03-28') {
                            $scope.isYinjian = true;   //中国银行业监督管理委员会
                        } else if (res.data.datafrom == 1) {
                            $scope.isBaojian = true;  //中国保险监督管理委员会
                        }
        
                    }
                    //wenzhang-content是否加上white-space
                    if (res.data.publishDate < '2019-09-01' || res.data.docUuid != null) {
                        $scope.isWhite_space = false;
                    }
                    else {
                        $scope.isWhite_space = true;
                    }
    
                    $("head > meta[name='ArticleTitle']").attr("content", res.data.docSubtitle);
                    if (res.data.publishDate != undefined) {
    
                        $("head > meta[name='PubDate']").attr("content", res.data.publishDate.substr(0, 16));
                    }
                    $("head > meta[name='ContentSource']").attr("content", res.data.docSource);
                })



                
                //  文章类型展示
                switch (res.data.documentType) {
                    case "0":
                        $scope.documentTypeDetail = "原创";
                        break;
                    case "1":
                        $scope.documentTypeDetail = "转载";
                        break;
                    case "2":
                        $scope.documentTypeDetail = "编译";
                        break;
                    default:
                        $scope.documentTypeDetail = "摘录";
                        break;
                }
              
                //面包屑
                var itemId = getParam("itemId");
                if (res.data.listTwoItem != undefined) {
                    for (var i = 0; i < res.data.listTwoItem.length; i++) {
                        for (var m = 0; m < res.data.listTwoItem[i].ItemLvs.length; m++) {
                            if (res.data.listTwoItem[i].ItemLvs[m] != null) {
                                if (res.data.listTwoItem[i].ItemID == itemId || res.data.listTwoItem[i].ItemLvs[m].itemId == itemId) {
                                    for (j = 0; j < res.data.listTwoItem[i].ItemLvs.length; j++) {
                                        res.data.listTwoItem[i].ItemLvs[j].itemPPid = res.data.listTwoItem[i].ItemLvs[1].itemPid;
                                        res.data.listTwoItem[i].ItemLvs[j].itemsubPId = res.data.listTwoItem[i].ItemLvs[1].itemId;
                                    }
                                    $scope.breadcrumb_detail = res.data.listTwoItem[i].ItemLvs;
                                    //文章所属栏目名、种类
                                    $("head > meta[name='ColumnName']").attr("content", $scope.breadcrumb_detail[$scope.breadcrumb_detail.length - 1].itemName);
                                    $("head > meta[name='ColumnType']").attr("content", $scope.breadcrumb_detail[$scope.breadcrumb_detail.length - 1].type);
                                    //判断是否是行政许可
                                    if ($scope.breadcrumb_detail[$scope.breadcrumb_detail.length - 1].itemName == "总局机关") {
                                        $scope.isXingzhengxuke = true;
                                    }
                                    // 判断政府信息公开页面中法律行政法规栏目数据的generaltype!=1，此栏目文章不在详情页显示来源
                                    if ($scope.breadcrumb_detail[$scope.breadcrumb_detail.length - 1].itemName == "法律行政法规") {
                                        $scope.showSource = false;
                                    }

                                    break;
                                }
                            }
                        }
                    }
                }
            }
            $(".content, .footer").show();
        }, function (res) {
            $(".content, .footer").show();

        }, 60 * 1000);
        $.ajax({
            type: 'get',
            url: '../../view/components/sts.html?' + (+ new Date),
            success: function (res) {

            }, 
            error: function (res) {

            }
        });
        //  文章pdf下载new
        // $scope.pdfFileDownload = function (docId) {
        //     if (!window.location.origin) {
        //         windowURL = window.location.protocol + "//" + window.location.hostname + (window.location.port ? ':' + window.location.port : '');
        //     } else {
        //         windowURL = window.location.origin;
        //     }
        //     window.open(windowURL + "/cbircweb/download/downloadPdf?docId=" + docId+"&zcfg=1");
        // }
        //  文章word下载new
        // $scope.docFileDownload = function (docId) {
        //     if (!window.location.origin) {
        //         windowURL = window.location.protocol + "//" + window.location.hostname + (window.location.port ? ':' + window.location.port : '');
        //     } else {
        //         windowURL = window.location.origin;
        //     }
        //     window.open(windowURL + "/cbircweb/download/downloadDoc?docId=" + docId);
        // }
        $scope.pdfFileDownload = function (docId) {
            // if (!window.location.origin) {
            //     windowURL = window.location.protocol + "//" + window.location.hostname + (window.location.port ? ':' + window.location.port : '');
            // } else {
            //     windowURL = window.location.origin;
            // }
            // var file_urlnew=windowURL + "/cbircweb/download/downloadPdf?docId=" + docId+"&zcfg=1";
            var file_urlnew = "/cbircweb/download/downloadPdf";
            $.ajax({
                url: file_urlnew,
                type: "GET",
                data: { docId: docId, zcfg: 1 },
                error: function (xhr, error, ex) {
                    if (xhr.status == '200') {
                        window.location.href = file_urlnew;
                    } else if (xhr.status == '404') {
                        alert("文件不存在！")
                    }
                },
                success: function () {
                    window.location.href = file_urlnew + "?docId=" + docId + "&zcfg=1";
                }
            });
        }
        //  文章word下载new
        $scope.docFileDownload = function (docId) {
            // if (!window.location.origin) {
            //     windowURL = window.location.protocol + "//" + window.location.hostname + (window.location.port ? ':' + window.location.port : '');
            // } else {
            //     windowURL = window.location.origin;
            // }
            var file_urlnew = "/cbircweb/download/downloadDoc";
            $.ajax({
                url: file_urlnew,
                type: "GET",
                data: { docId: docId ,zcfg:1,itemId:itemId},
                error: function (xhr, error, ex) {
                    if (xhr.status == '200') {
                        window.location.href = file_urlnew;
                    } else if (xhr.status == '404') {
                        alert("文件不存在！")
                    }else if (xhr.status == '423') {
                        alert("文件尚未生成！")
                    }
                },
                success: function () {
                    window.location.href = file_urlnew + "?docId=" + docId+ "&zcfg=1";
                }
            });
        }

        //判断url是否是404
        $scope.fileDownload = function (file_url) {
            $.ajax({
                url: file_url,
                error: function (xhr, error, ex) {
                    if (xhr.status == '200') {
                        window.location.href = file_url;
                    } else if (xhr.status == '404') {
                        alert("文件不存在！")
                    }
                },
                success: function () {
                    window.location.href = file_url;
                }
            });
        }
    });

    app.controller('itemDetailRedCtrl', function ($scope, global) {
        $scope.docId = getParam("docId");
        $scope.itemId = getParam("itemId");
    });


    app.controller('rulesDetile', function ($scope, global,$rootScope,$timeout) {
      var docIdd = getParam("docId");
      global.getCDN({ url: '/DocInfo/SelectByDocId', params: { docId: docIdd } }, function (data) {
        if (data.rptCode == 200) {
            $scope.data = data.data
            $scope.docTitle = data.data.docTitle
            $scope.caption = data.data.caption
            $scope.texthtml = data.data.docClob;

            // 规章页面 meta数据补全
            $("head > meta[name='ArticleTitle']").attr("content", data.data.docTitle);
            if (data.data.publishDate != undefined) {

                $("head > meta[name='PubDate']").attr("content", data.data.publishDate.substr(0, 16));
            }
            $("head > meta[name='ContentSource']").attr("content", data.data.docSource);
            
            $timeout(function () {
              var pArr = $('p>span');
              for(var i=0;i<pArr.length;i++) {
               if($(pArr[i]).length == 1 && ($(pArr[i])[0].innerHTML == '<br>' || $(pArr[i])[0].innerHTML == '<br />' ||$(pArr[i])[0].innerHTML == '<br/>' ||$(pArr[i])[0].innerHTML == '<BR>')){
                 $(pArr[i]).addClass('zh-child')
               }
              }
            },60);
            if (data.data.remark2) {
              $scope.valueArr = data.data.remark2.match(/\[[^\]]+\]\s*[(|（][^)^）^(^（)]+[)|）]\s*/g)// 附录
              var regExp2 = /\[[^\]]+\]\s*[(|（][^)^）^(^（)]+[)|）]\s*/g;

              if (regExp2.test(data.data.remark2) == false) {

                data.data.rules = "1";
              } else {
                data.data.rules = "0";

                  $scope.editSubContent = []
                  for (var i = 0; i < $scope.valueArr.length; i++) {
                      $scope.valueArr[i] = $scope.valueArr[i].replace(/(^\s*)|(\s*$)/g);
                      //解析存入的[标题](地址)

                      $scope.valueArr[i].replace(/\[(.*)\]\s*[\(|（](.*)[\)|）]/, function () {
                          $scope.editSubContent.push({ title: arguments[1], href: arguments[2] })
                          // console.log(arguments)
                      })
                  }

              }

          } 
        } else {
            console.log(data);
        }
        
         // 从接口获取挂牌时间
         var sysBuildDate = '';
         global.getCDN({ url: '/Skin/getSysParams' }, function (re) {
            if (re.rptCode == 200) {
                sysBuildDate = re.data.ceremony_time;
            }
            // 如果发文日期不早于挂牌时间 规章红头展示“国家金融监督管理总局规章” 其他逻辑不变
            if(new Date(data.data.builddate) >= new Date(sysBuildDate)) {
                $scope.rulesTitle = '国家金融监督管理总局';
            } else {
                $scope.rulesTitle = '中国银行保险监督管理委员会';
            }
         })

         $(".content, .footer").show();


        }, function (data) {
            console.log(data);
        });
    $scope.itemId = getParam("itemId");
    $scope.rulesDocFileDownload = function (type) {
      // if(itemId==927){
      //     var zcfg=0;
      // }else{
      //     var zcfg=1;
      // }
      var file_urlnew = type ==1? "/cbircweb/download/downloadguizDoc" : "/cbircweb/download/downloadguizPdf";
      // var file_urlnew2=windowURL + "/cbircweb/download/downloadPdf?docId=" + docId+"&zcfg=1";
      $.ajax({
          url: file_urlnew,
          type:"GET",
          data:{docId:docIdd },
          error: function (xhr, error, ex) {
              if (xhr.status == '200') {
                  window.location.href = file_urlnew;
              } else if (xhr.status == '404') {
                  alert("文件不存在！")
              }else if (xhr.status == '423') {
                  alert("文件尚未生成！")
              }
          },
          success: function () {
              window.location.href = file_urlnew+ "?docId=" + docIdd;
          }
      });
  }
  });
})(app);