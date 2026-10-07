//接口和json文件地址
var originUrl = location.origin;
if(!originUrl) originUrl = window.location.protocol + "//" + window.location.hostname + (window.location.port ? ':' + window.location.port : '');
var apiUrl_dev = originUrl + "/cbircweb";
var apiUrl_dev_zwzx = originUrl + "/cbirczwzxweb";
var apiUrl_cdn =  originUrl + "/cn/static/data";
// if (isBig5()) {
//     if (isHttps()) {
//         var apiUrl_dev = "https://big5.cbirc.gov.cn/cbircweb";
//         var apiUrl_dev_zwzx = "https://big5.cbirc.gov.cn/cbirczwzxweb";
//         var apiUrl_cdn = "https://big5.cbirc.gov.cn/cn/static/data";
//     } else {
//         var apiUrl_dev = "http://big5.cbirc.gov.cn/cbircweb";
//         var apiUrl_dev_zwzx = "http://big5.cbirc.gov.cn/cbirczwzxweb";
//         var apiUrl_cdn = "http://big5.cbirc.gov.cn/cn/static/data";
//     }
// } else {
//     if (isHttps()) {
//         var apiUrl_dev = "https://www.cbirc.gov.cn/cbircweb";
//         var apiUrl_dev_zwzx = "https://www.cbirc.gov.cn/cbirczwzxweb";
//         var apiUrl_cdn = "https://www.cbirc.gov.cn/cn/static/data";
//     } else {
//         var apiUrl_dev = "http://www.cbirc.gov.cn/cbircweb";
//         var apiUrl_dev_zwzx = "http://www.cbirc.gov.cn/cbirczwzxweb";
//         var apiUrl_cdn = "http://www.cbirc.gov.cn/cn/static/data";
//     }
// }

var sysId = '20001'    // 系统ID 5位长度
var channelType = (navigator.userAgent.toLocaleLowerCase().indexOf('android') > -1 || navigator.userAgent.toLocaleLowerCase().indexOf('iphone') > -1) ? '01' : '02' // 渠道类型 01,移动端;02，PC端;03，P5;04,其它渠道
var serialNo = '1799'   // 序列号
var machineNo = 'PC90' // 机器号
var CTenancyIi = '005500000000'
var decodeKey = '636269726332303230303932474F566D';
var isRequestSkinJson = false; // 是否请求皮肤参数

var app = angular.module('myApp', []);

(function ($) {
    $.ajaxSettings.async = false;
    $.fn.apiUrl_dev = apiUrl_dev;
    $.fn.apiUrl_dev_zwzx = apiUrl_dev_zwzx;
    $.fn.sysId = sysId
    $.fn.channelType = channelType
    $.fn.serialNo = serialNo
    $.fn.machineNo = machineNo
    $.fn.CTenancyIi = CTenancyIi

    /* 动态设置meta标签 */
    $("head > meta[name='author']").attr("content", "国家金融监督管理总局");
    $("head > meta[name='description']").attr("content", "国家金融监督管理总局");
    $("head > meta[name='keywords']").attr("content", "国家金融监督管理总局");
    //$("head > meta[name='viewport']").attr("content", "width=device-width, initial-scale=1");
    $("head > meta[name='viewport']").attr("content", "width=1100");
    $("head > meta[name='SiteName']").attr("content", "国家金融监督管理总局");
    // if (isBig5()) {
    //     $("head > meta[name='SiteDomain']").attr("content", "big5.cbirc.gov.cn");
    // } else {
    //     $("head > meta[name='SiteDomain']").attr("content", "www.cbirc.gov.cn");
    // }
    $("head > meta[name='SiteDomain']").attr("content", location.host);
    $("head > meta[name='SiteIDCode']").attr("content", "bm55000001");
    //$("head > link[rel='Shortcut Icon']").attr("href", "/cn/static/images/common/guohui.ico");
    //$("head > title").html("国家金融监督管理总局");

    //ie8兼容console
    window.console = window.console || (function () {
        var c = {};
        c.log = c.warn = c.debug = c.info = c.error = c.time = c.dir = c.profile = c.clear = c.exception = c.trace = c.assert = function () { };
        return c;
    })();

    /* 加载碎片组件 */
    var tplList = $("tpl");
    for (var i = 0; i < tplList.length; i++) {
        var tpl = $(tplList[i]);
        var active = tpl.attr("active");
        var active2 = tpl.attr("active2");
        var src = tpl.attr("src");
        $.get(src, function (data) {
            // /*左侧一级菜单文字颜色*/
            if (active != undefined) {
                var li = $(data).find("#" + active).addClass("active");
                data = li.parent().parent();
            }
            // /*左侧二级菜单文字颜色*/
            if (active2 != undefined) {
                var li2 = $(data).find("#" + active2).addClass("active");
                var li3 = li2.parent().addClass("caidan-left-erji-active");
                data = li3.parent().parent().parent();
            }
            /*tpl文件替换内容*/
            tpl.replaceWith(data);
        });
    }

    $.ajaxSettings.async = true;
    $(".header-menu > a").click(function () {
        $(".header-menu").css("display", "none");
        $(".header-menu-close").css("display", "block");
        $(".header-right2").css("display", "block");
    });
    $(".header-menu-close > a").click(function () {
        $(".header-menu").css("display", "block");
        $(".header-menu-close").css("display", "none");
        $(".header-right2").css("display", "none");
    });
    $("[list-panel-id]").each(function () {
        var panelid = $(this).attr("list-panel-id");
        $(this).parent().mouseenter(function () {
            $(this).parent().children(".tab").each(function () {
                $(this).removeClass("active");
            });
            $(this).addClass("active");
            $("#" + panelid).parent().children(".panel").each(function () {
                $(this).removeClass("active");
            });
            $("#" + panelid).addClass("active");
        });

    });
    var isMouseOn = 0;
})($);

/*打印*/
function printpage() {
    window.print();
}

/*分享-更多*/
$.fn.myHoverTip = function (divId) {
    $('#share-more-all').hide();
    var div = $("#" + divId); //要浮动在这个元素旁边的层
    div.css("position", "absolute");//让这个层可以绝对定位
    var self = $(this); //当前对象
    self.hover(function () {
        div.css("display", "block");
        var p = self.position(); //获取这个元素的left和top
        var x = p.left + self.width() - 37;//获取这个浮动层的left
        var docWidth = $(document).width();//获取网页的宽
        if (x > docWidth - div.width() - 20) {
            x = p.left - div.width();
        }
        div.css("left", x);
        div.css("top", p.top + 18);
        div.show();
    },
        function () {
            div.css("display", "none");
        });
    return this;
}

// function getParam(name) {
//     var reg = new RegExp("(^|&)" + name + "=([^&]*)(&|$)");
//     var r = window.location.search.substr(1).match(reg);
//     if (r != null) return unescape(r[2]); return null;
// };

//获取地址栏参数支持中英文
function getParam(name,search) {
    // 获取参数
    var url = window.location.search;
    // 正则筛选地址栏
    var reg = new RegExp("(^|&)" + name + "=([^&]*)(&|$)");
    // 匹配目标参数
    var result = url.substr(1).match(reg);
    // return result ? decodeURIComponent(result[2]) : null;

    // 地址栏出现单独的%作为value值，则将其转换为%25
    try {
        return result ? decodeURIComponent(result[2]) : null;
    } catch (error) {
        var tempIndexArr = [];
        for(var i = 0; i < url.length; i++) {
            if(url[i] == "%") {
                if(url[i+1] && url[i+2] && url[i+1] == '2' && url[i+2] == '5') {
                    continue;
                } else {
                    tempIndexArr.push(i)
                }
            } 
        }
        var init1 = 0;
        var finalRes = "";
        for(var j = 0; j < tempIndexArr.length; j++) {
            finalRes += url.substring(init1,tempIndexArr[j]).concat("%25");
            init1 = tempIndexArr[j] + 1;
            if(j == tempIndexArr.length - 1)  finalRes += url.substring(init1, url.length);
        }
        if(finalRes) {
            window.location.search = finalRes;
            result = url.substr(1).match(reg);
            return result ? decodeURIComponent(result[2]) : null;
        }
    }
    // try {
    //   return result ? decodeURIComponent(result[2]) : null;
    // } catch (error) {
    //   if(!!window.ActiveXObject || 'ActiveXObject' in window ){
    //     return result[2]
    //   }else {
    //     for(var i=0;i<result[2].length;i++){
    //       if(result[2][length] == '%') {
    //         result[2] = result[2].substring(0,result[2].length-1)
    //         if(result[2].charAt(result[2].length-1) != '%'){
    //           return decodeURIComponent(result[2])
    //         }
    //       }else {
    //         return result[2]
    //       }
    //     }
    //   }
    // }
};
function getParam2(name) {
    var p = location.hash.replace("#", "");
    return p == "" ? 1 : p * 1;
};
function exp2019(data){
    for( var j=0,len = data.length;j<len;j++){
        var matchData0 = data[j].docSubtitle.replace(/[^0-9]+/g, "");
        data[j].year = matchData0;
        // debugger
        if(data[j].year!='' && (data[j].year-0)>2019){
            data[j].isgtyear=true;
            //  return isgtyear=true;
        }else{
            data[j].isgtyear=false;
            //  return isgtyear=false;
        }
    
    }
    return data;
}
//判断是否为繁体网站
function isBig5() {
    if (location.href.toLowerCase().indexOf("big5") > -1) {
        return true;
    } else {
        return false;
    }
}

//判断是否为https
function isHttps() {
    if (location.href.toLowerCase().indexOf("https") > -1) {
        return true;
    } else {
        return false;
    }
}

app.config(function ($provide) {
    $provide.provider('global', function () {
        this.$get = function ($http, $rootScope, $timeout) {
            // 常量数据
            // 20大栏目ID
            $rootScope.TWENTIETH_CONGRESS_ID = '4226';
            
            var global = {
                //开发接口地址
                apiUrl_dev: apiUrl_dev,
                apiUrl_dev_zwzx: apiUrl_dev_zwzx,
                apiUrl_cdn: apiUrl_cdn,
                decodeKey: decodeKey,
                isRequestSkinJson: isRequestSkinJson
            };
            global.allKeys = [];
            global.now = function () {
                var d = new Date();
                var month = (d.getMonth() + 1) < 10 ? '0' + (d.getMonth() + 1) : (d.getMonth() + 1);
                var date = d.getDate() < 10 ? '0' + d.getDate() : d.getDate();
                var hours = d.getHours() < 10 ? '0' + d.getHours() : d.getHours();
                var minutes = d.getMinutes() < 10 ? '0' + d.getMinutes() : d.getMinutes();
                var seconds = d.getSeconds() < 10 ? '0' + d.getSeconds() : d.getSeconds();

                return d.getFullYear() + '-' + month + '-' + date + ' ' + hours + ':' + minutes + ':' + seconds;
            }
            global.get = function (api, callback) {
                $.ajax({
                    type: "GET",
                    contentType: "text/plain",
                    url: api.url,
                    data: api.data,

                    headers: {

                        'Authorization': 'Bearer:' + $.cookie('token') + ';' + $.cookie('refreshToken')
                    },
                    success: function (data) {
                        if (data.rptCode == 10007 || data.rptCode == 10011|| data.rptCode == 10012) { // 未登录
                            $.removeCookie('refreshToken', { path: '/' })
                            $.removeCookie('token', { path: '/' })
                            $.removeCookie('userId', { path: '/' })
                            $.removeCookie('name', { path: '/' })
                            $.removeCookie('usercode', { path: '/' })
                            window.location.reload()
                            // window.location.href = "/cn/view/pages/index/login.html"
                            window.location.href = '/cn/view/pages/index/login.html'
                            return false
                        } else if (data.rptCode == 10010) { // token过期需要重新获取 && 重新请求
                            global.post({
                                url: global.apiUrl_dev + '/WebUserInfo/refreshAuthorization'
                            }, function (res) {
                                if (res.rptCode == 200) {
                                    $rootScope.token = res.data.token
                                    $rootScope.refreshToken = res.data.refreshToken
                                    $.cookie('token', res.data.token, { path: '/' })
                                    $.cookie('refreshToken', res.data.refreshToken, { path: '/' })
                                    global.get(api, callback)
                                    return false
                                }
                            })
                        }
                        callback(data)
                    }
                });
            }
            global.post = function (api, callback, errorCallBack) {
                $.ajax({
                    type: "post",
                    contentType: "text/plain",
                    url: api.url,
                    data: JSON.stringify(api.data),
                    headers: {
                        'Authorization': 'Bearer:' + $.cookie('token') + ';' + $.cookie('refreshToken')
                    },
                    success: function (data) {
                        // debugger
                        if (data.rptCode == 10007 || data.rptCode == 10011) {
                            if(data.rptCode == 10011){
                                // alert("返回的本来是：登录已过期，请重新登录")
                                alert("登录超时，请重新登录");
                                $.removeCookie('refreshToken', { path: '/' })
                                $.removeCookie('token', { path: '/' })
                                $.removeCookie('userId', { path: '/' })
                                $.removeCookie('name', { path: '/' })
                                $.removeCookie('usercode', { path: '/' })
                                // window.location.reload()
                                window.location.href = "/cn/view/pages/ItemList.html?itemPId=945&itemId=947&itemUrl=hudongjiaoliu/woyaozixun.html&itemName=我要咨询"
                                return false;
                            }else{
                                $.removeCookie('refreshToken', { path: '/' })
                                $.removeCookie('token', { path: '/' })
                                $.removeCookie('userId', { path: '/' })
                                $.removeCookie('name', { path: '/' })
                                $.removeCookie('usercode', { path: '/' })
                                // window.location.reload()
                                window.location.href = "/cn/view/pages/index/login.html"
                                return false;              
                                }
                             // 未登录
                          
                        } 
                        else if (data.rptCode == 10010) { // token过期需要重新获取 && 重新请求
                            global.post({
                                url: global.apiUrl_dev + '/WebUserInfo/refreshAuthorization'
                            }, function (data) {
                                // alert(data)
                                if (data.rptCode == 200) {
                                    $rootScope.token = data.data.token ? data.data.token : ''
                                    $rootScope.refreshToken = data.data.refreshToken ? data.data.refreshToken : ''

                                    data.data.token ? $.cookie('token', data.data.token, { path: '/' }) : ''
                                    data.data.refreshToken ? $.cookie('refreshToken', data.data.refreshToken, { path: '/' }) : ''

                                    global.post(api, callback)
                                    return false
                                }
                            })
                        } else if (data.rptCode == 200) {
                          $rootScope.token = data.data.token ? data.data.token : ''
                            $rootScope.refreshToken = data.data.refreshToken ? data.data.refreshToken : ''

                            data.data.token ? $.cookie('token', data.data.token, { path: '/' }) : ''
                            data.data.refreshToken ? $.cookie('refreshToken', data.data.refreshToken, { path: '/' }) : ''
                    }
                        callback(data)
                    },
                    error: function (data) {
                        // debugger
                        errorCallBack(data)
                    }
                })
            }
            global.getCDN = function (api, success, error, localCacheTS) {
                this.ajaxCDN2(api, success, error, 'get', localCacheTS);
            }
            global.postCDN = function (api, success, error, localCacheTS) {
                this.ajaxCDN(api, success, error, 'post', localCacheTS);
            }
            global.ajaxCDN = function (api, success, error, method, localCacheTS) {
                var dynamicUrl = this.apiUrl_dev;
                var pa = [];
                var npa = [];
                var ps = "";
                if (api.params != undefined) {
                    for (var key in api.params) {
                        pa.push(key);
                    }
                }

                if (pa.length > 0) {
                    pa.sort();
                    for (var i = 0; i < pa.length; i++) {
                        npa.push(pa[i] + "=" + encodeURIComponent(api.params[pa[i]]));
                    }
                    ps = "[" + npa.join() + "]";
                }
                var key = encodeURI(this.apiUrl_cdn + "/" + api.url.substr(1, api.url.length - 1).replace("/", "_") + ps + '.json');
                var tsKey = key + "_timestamp";
                var cacheData = sessionStorage.getItem(key);
                var ts = sessionStorage.getItem(tsKey);
                // 调用统一接口参数相同，时间间隔小于10秒从sessionstorage取数据
                var timeSpan = new Date - new Date(ts * 1);
                var defaultTS = 10000;
                if (localCacheTS != undefined) {
                    defaultTS = localCacheTS;
                }
                if (cacheData != undefined && ts != undefined && timeSpan <= defaultTS) {
                    success(JSON.parse(cacheData));
                } else {
                    // 从cdn取
                    if (this.apiUrl_cdn != "" && (global.getPageIndex2() <= 3)) {

                        $http({
                            method: method,
                            url: key
                        }).success(function (res) {
                            if (success != undefined) {
                                // 缓存到sessionstorage
                                sessionStorage.setItem(key, JSON.stringify(res));
                                sessionStorage.setItem(tsKey, + new Date);
                                if (success != undefined) {
                                    success(res);
                                }
                            }
                        }).error(function (res) {
                            $http({
                                method: method,
                                url: dynamicUrl + api.url,
                                params: api.params
                            }).success(function (res) {
                                sessionStorage.setItem(key, JSON.stringify(res));
                                sessionStorage.setItem(tsKey, + new Date);
                                if (success != undefined) {
                                    success(res);
                                }
                            }).error(function (res) {
                                if (error != undefined) {
                                    error(res);
                                }
                            })
                        })

                    } else {
                        // 从接口取
                        $http({
                            method: method,
                            url: this.apiUrl_dev + api.url,
                            params: api.params
                        }).success(function (res) {
                            sessionStorage.setItem(key, JSON.stringify(res));
                            sessionStorage.setItem(tsKey, + new Date);
                            if (success != undefined) {
                                success(res);
                            }
                        }).error(function (res) {
                            if (error != undefined) {
                                error(res);
                            }
                        })
                    }
                }

            }
            global.ajaxCDN2 = function (api, success, error, method, localCacheTS) {
                if (sessionStorage.getItem("currentUrl") != location.href) {
                    var noClear = JSON.parse(sessionStorage.getItem("_noclear"));
                    sessionStorage.clear();
                    if (noClear != undefined) {
                        for (var i = 0; i < noClear.length; i++) {
                            sessionStorage.setItem(noClear[i].key, noClear[i].value);
                        }
                    }
                    else {
                        noClear = [];
                    }
                    sessionStorage.setItem("_noclear", JSON.stringify(noClear));
                    sessionStorage.setItem("currentUrl", location.href);
                }
                var dynamicUrl = this.apiUrl_dev;
                //console.log(method, " ", dynamicUrl + api.url);
                var pa = [];
                var npa = [];
                var ps = "";
                if (api.params != undefined) {
                    for (var key in api.params) {
                        pa.push(key);
                    }
                }
                if (pa.length > 0) {
                    pa.sort();
                    for (var i = 0; i < pa.length; i++) {
                        npa.push(pa[i] + "=" + encodeURIComponent(api.params[pa[i]]));
                    }
                    //ps = "["+npa.join()+"]";
                    //ps = "__"+npa.join();
                    ps = npa.join();
                }

                //var key = encodeURI(this.apiUrl_cdn + "/" +  api.url.substr(1,api.url.length-1).replace("/","_") + ps + '.json');

                var key = encodeURI(api.url + '/data_' + ps + '.json');
                var big5Key = encodeURI('/big5' + api.url + '/data_' + ps + '.json');
                var isDuplicate = false;
                for (var i = 0; i < global.allKeys.length; i++) {
                    if (global.allKeys[i].key == key && new Date - new Date(global.allKeys[i].ts * 1) <= 1000) {
                        isDuplicate = true;
                        // console.log("duplicate call ", key);
                    }
                }
                if (!isDuplicate) {
                    global.allKeys.push({ key: key, ts: +new Date });
                    sessionStorage.setItem("allKeys", JSON.stringify(global.allKeys));
                }
                var tsKey = key + "_timestamp";
                var cacheData = undefined;
                var isBig5 = global.isBig5();
                if (isBig5) {
                    cacheData = sessionStorage.getItem(big5Key); // 繁体数据
                } else {
                    cacheData = sessionStorage.getItem(key); // 简体数据
                }

                var ts = sessionStorage.getItem(tsKey);
                // 调用统一接口参数相同，时间间隔小于10秒从sessionstorage取数据
                var timeSpan = new Date - new Date(ts * 1);
                var defaultTS = 10000;
                if (localCacheTS != undefined) {
                    defaultTS = localCacheTS;
                }
                if (cacheData != undefined && ts != undefined && timeSpan <= defaultTS) {
                    //$rootScope.$apply(function(){
                    success(JSON.parse(cacheData));
                    //});

                } else {
                    // 从cdn取
                    if (this.apiUrl_cdn != "" && (global.getPageIndex2() <= 3)) {
                        // console.log("cdn ", key);
                        $.ajax({
                            type: method,
                            url: this.apiUrl_cdn + key,
                            success: function (res) {
                                // console.log("got from cdn,", key);
                                if (success != undefined) {
                                    // 缓存到sessionstorage
                                    if (isBig5) {
                                        sessionStorage.setItem(big5Key, JSON.stringify(res));
                                    } else {
                                        sessionStorage.setItem(key, JSON.stringify(res));
                                    }
                                    sessionStorage.setItem(tsKey, + new Date);
                                    if (success != undefined) {
                                        $rootScope.$apply(function () {
                                            success(res);
                                        });
                                    }
                                }
                            },
                            error: function (res) {
                                $.ajax({
                                    type: method,
                                    url: dynamicUrl + api.url,
                                    data: api.params,
                                    success: function (res2) {
                                        if (isBig5) {
                                            sessionStorage.setItem(big5Key, JSON.stringify(res2));
                                        } else {
                                            sessionStorage.setItem(key, JSON.stringify(res2));
                                        }
                                        sessionStorage.setItem(tsKey, + new Date);
                                        $rootScope.$apply(function () {
                                            success(res2);
                                        });
                                    },
                                    error: function (res2) {
                                        if (error != undefined) {
                                            error(res2);
                                        }
                                    }
                                });
                            }
                        });
                    } else {
                        // 从接口取
                        // console.log("cdn config failure ");
                        //console.log("api ", dynamicUrl + api.url);
                        $.ajax({
                            type: method,
                            url: dynamicUrl + api.url,
                            data: api.params,
                            success: function (res) {
                                if (isBig5) {
                                    sessionStorage.setItem(big5Key, JSON.stringify(res));
                                } else {
                                    sessionStorage.setItem(key, JSON.stringify(res));
                                }
                                sessionStorage.setItem(tsKey, + new Date);
                                if (success != undefined) {
                                    $rootScope.$apply(function () {
                                        success(res);
                                    });

                                }
                            },
                            error: function (res) {
                                if (error != undefined) {
                                    error(res);
                                }
                            }
                        });
                    }
                }

            }
            global.getCDNFileName = function (api) {
                var ps = "";
                var pa = [];
                var npa = [];
                if (api.params != undefined) {
                    for (var key in api.params) {
                        pa.push(key);
                    }
                }
                if (pa.length > 0) {
                    pa.sort();
                    for (var i = 0; i < pa.length; i++) {
                        npa.push(pa[i] + "=" + encodeURIComponent(api.params[pa[i]]));
                    }
                    ps = npa.join();
                }

                var key = encodeURI(api.url + '/data_' + ps + '.json');
                return this.apiUrl_cdn + key
            }
            // global.toast = function (msg, width, height, duration) {
            //     if (width == undefined) {
            //         width = 100;
            //     }
            //     if (height == undefined) {
            //         height = 100;
            //     }
            //     var modalHtml = "<div class='ybj-toast' style='width:" + width + "px;height:" + height + "px;'><div><div>" + msg + "</div></div></div>";
            //     $("body").append(modalHtml);
            //     var timeout = 1000;
            //     if (duration != undefined) {
            //         timeout = duration;
            //     }
            //     $timeout(function () {
            //         $(".ybj-toast").remove();
            //     }, timeout);
            //     //modal.find(".loading").remove();
            // }
            var isclick = true;
            global.toast = function (msg, width, height, duration) {
                if (width == undefined) {
                    width = 100;
                }
                if (height == undefined) {
                    height = 100;
                }
                var modalHtml = "<div class='ybj-toast' style='width:" + width + "px;height:" + height + "px;'><div><div>" + msg + "</div></div></div>";
                if (isclick) {
                    isclick = false;
                    $("body").append(modalHtml);
                }
                var timeout = 1000;
                if (duration != undefined) {
                    timeout = duration;
                }
                $timeout(function () {
                    $(".ybj-toast").remove();
                    isclick = true;
                }, timeout);
                //modal.find(".loading").remove();
            }
            global.getPageIndex = function () {
                var p = getParam("p");
                return p == null ? 1 : p * 1;
            }
            global.getPageIndex2 = function () {
                var p = location.hash.replace("#", "");
                if(location.hash.indexOf('location') > -1){  // 互动交流-政务咨询-信访投诉进入联系方式，location静态化问题修复
                    p = "";
                }
                return p == "" ? 1 : p * 1;

            }

            //sessionStorage支持繁简体
            global.sessionStorage = {};
            global.sessionStorage.getItem = function (itemName) {
                var itemList = undefined;
                if (global.isBig5()) {
                    itemList = sessionStorage.getItem("/big5/" + itemName); // 繁体数据
                } else {
                    itemList = sessionStorage.getItem(itemName); // 简体数据
                }
                return itemList;
            }
            global.sessionStorage.setItem = function (itemName, data, isClear) {
                if (global.isBig5()) {
                    var key = "/big5/" + itemName;
                    sessionStorage.setItem(key, data); // 繁体数据
                    if (isClear == false) {
                        var noClear = JSON.parse(sessionStorage.getItem("_noclear"));
                        var hasValue = false;
                        if (noClear == undefined || noClear == null) {
                            noClear = [];
                        }
                        for (var i = 0; i < noClear.length; i++) {
                            if (noClear[i].key == key) {
                                noClear[i].value = data;
                                hasValue = true;
                                break;
                            }
                        }
                        if (!hasValue) {
                            noClear.push({ key: key, value: data });
                        }
                        sessionStorage.setItem("_noclear", JSON.stringify(noClear));
                    }
                } else {
                    sessionStorage.setItem(itemName, data); // 简体数据
                    var key = itemName;
                    if (isClear == false) {
                        var noClear = JSON.parse(sessionStorage.getItem("_noclear"));
                        var hasValue = false;
                        if (noClear == undefined || noClear == null) {
                            noClear = [];
                        }
                        for (var i = 0; i < noClear.length; i++) {
                            if (noClear[i].key == key) {
                                noClear[i].value = data;
                                hasValue = true;
                                break;
                            }
                        }
                        if (!hasValue) {
                            noClear.push({ key: key, value: data });
                        }
                        sessionStorage.setItem("_noclear", JSON.stringify(noClear));
                    }
                }
            }

            //localStorage支持繁简体
            // global.localStorage = {};
            // global.localStorage.getItem = function (itemName) {
            //     var itemList = undefined;
            //     if (global.isBig5()) {
            //         itemList = localStorage.getItem("/big5/" + itemName); // 繁体数据
            //     } else {
            //         itemList = localStorage.getItem(itemName); // 简体数据
            //     }
            //     return itemList;
            // }
            // global.localStorage.setItem = function (itemName, data) {
            //     if (global.isBig5()) {
            //         itemList = localStorage.setItem("/big5/" + itemName, data); // 繁体数据
            //     } else {
            //         itemList = localStorage.setItem(itemName, data); // 简体数据
            //     }
            //     return itemList;
            // }
            // global.localStorage.removeItem = function (itemName) {
            //     if (global.isBig5()) {
            //         localStorage.removeItem("/big5/" + itemName); // 清除繁体数据
            //     } else {
            //         localStorage.removeItem(itemName); // 清除简体数据
            //     }
            // }

            global.getItemListByName = function (data, itemName) {
                for (var i = 0; i < data.length; i++) {
                    if (data[i].itemName == itemName) {
                        return data[i];
                    }
                }
            }
            global.getItemIdByName = function (data, itemName) {
                for (var i = 0; i < data.length; i++) {
                    if (data[i].itemName == itemName) {
                        return data[i].itemId;
                    }
                }
            }
            global.getItemNameById = function (data, itemId) {
                for (var i = 0; i < data.length; i++) {
                    if (data[i].itemId == itemId) {
                        return data[i].itemName;
                    }
                }
            }
            // 一级
            global.getItemNameById2 = function (itemId, level) {
                var itemList = global.sessionStorage.getItem("itemList");
                if (itemList != undefined) {
                    var itemListJson = JSON.parse(itemList);
                    for (var i = 0; i < itemListJson.length; i++) {
                        if (itemListJson[i].itemId == itemId) {
                            return itemListJson[i].itemName;
                        }
                    }
                } else {
                    global.getCDN({
                        url: '/item/getWebMenuItem',
                        params: { lang: "CN" }
                    }, function (res) {
                        if (res.rptCode == 200) {
                            global.sessionStorage.setItem("itemList", JSON.stringify(res.data), false);
                            var itemListJson = res.data;
                            for (var i = 0; i < itemListJson.length; i++) {
                                if (itemListJson[i].itemId == itemId) {
                                    return itemListJson[i].itemName;
                                }
                            }
                        }
                    });
                }
            }
            global.getItemIdByName2 = function (itemName, level, success) {
                var itemList = global.sessionStorage.getItem("itemList");
                if (itemList != undefined) {
                    var itemListJson = JSON.parse(itemList);
                    for (var i = 0; i < itemListJson.length; i++) {
                        if (itemListJson[i].itemName == itemName) {
                            if (success != undefined) {
                                success(itemListJson[i].itemId);
                            }
                            break;
                        }
                    }
                } else {
                    global.getCDN({
                        url: '/item/getWebMenuItem',
                        params: { lang: "CN" }
                    },
                        function (res) {
                            if (res.rptCode == 200) {
                                //console.log(res);
                                // 缓存栏目数据
                                global.sessionStorage.setItem("itemList", JSON.stringify(res.data), false);
                                var itemListJson = res.data;
                                for (var i = 0; i < itemListJson.length; i++) {
                                    if (itemListJson[i].itemName == itemName) {
                                        if (success != undefined) {
                                            success(itemListJson[i].itemId);
                                        }
                                        break;
                                    }
                                }
                            }
                        });
                }
            }

            global.parseItemUrl2 = function (item, pid) {
                var prefix = "/cn/view/pages/"
                if (item != undefined) {
                    if (item.itemUrl != undefined) {
                        // 相对路径
                        if (item.itemUrl.toLowerCase().substring(0, 8).indexOf('relative') > -1) {
                            return prefix + item.itemUrl.toLowerCase().replace('relative:', '');
                        }
                        // 不能点击 
                        else if (item.itemUrl.toLowerCase().substring(0, 7).indexOf('noclick') > -1) {
                            return '';
                        }
                        // 绝对路径 
                        else if (item.itemUrl.toLowerCase().substring(0, 4).indexOf('http') > -1) {
                            return item.itemUrl;
                        }
                        // 点击左侧一级跳到第一个二级菜单
                        else if (item.itemUrl.toLowerCase().substring(0, 9).indexOf('tosubmenu') > -1) {
                            return prefix + 'ItemList.html?itemPId=' + item.itemPid + '&itemId=' + item.subItemslist[0].itemId + '&itemUrl=' + item.subItemslist[0].itemUrl + '&itemName=' + item.subItemslist[0].itemName;
                        } else {
                            if (pid == undefined) {
                                return prefix + 'ItemList.html?itemPId=' + item.itemPid + '&itemId=' + item.itemId + '&itemUrl=' + item.itemUrl + '&itemName=' + item.itemName;
                            } else {
                                return prefix + 'ItemList.html?itemPId=' + pid + '&itemId=' + item.itemId + '&itemUrl=' + item.itemUrl + '&itemName=' + item.itemName;
                            }
                        }
                    }

                } else {
                    return '';
                }
            }

            //文章列表加入itemId
            global.parseItemDocInfoVOList = function (item, itemId, count) {
                if (item != undefined) {
                    if (count > item.docInfoVOList.length) {
                        count = item.docInfoVOList.length;
                    }
                    for (var i = 0; i < count; i++) {
                        item.docInfoVOList[i].itemId = itemId;
                    }
                    return item.docInfoVOList.slice(0, count);
                } else {
                    return '';
                }
            }

            //文章图片列表加入itemId
            global.parseItemDocImageInfoVOList = function (item, itemId, count) {
                if (item != undefined) {
                    if (count > item.docImageInfoVOList.length) {
                        count = item.docImageInfoVOList.length;
                    }
                    for (var i = 0; i < count; i++) {
                        item.docImageInfoVOList[i].itemId = itemId;
                    }
                    return item.docImageInfoVOList.slice(0, count);
                } else {
                    return '';
                }
            }

            //判断是否是繁体网站big5
            global.isBig5 = function () {
                if (location.href.toLowerCase().indexOf("big5") > -1) {
                    return true;
                } else {
                    return false;
                }
            }
            global.getIEVersion = function () {
                var ua = navigator.userAgent.toLowerCase();
                var isIE = ua.indexOf("compatible") > -1 && ua.indexOf("msie") > -1; // if ie < 11
                var isEdge = ua.indexOf("edge") > -1 && !isIE; // if IE Edge
                var isIE11 = ua.indexOf("trident") > -1 && ua.indexOf("rv:11.0") > -1;
                if (isIE) {
                    var reIE = new RegExp("msie (\\d+\\.\\d+);");
                    reIE.test(ua);
                    var fIEVersion = parseFloat(RegExp["$1"]);
                    if (fIEVersion == 7) {
                        return 7;
                    } else if (fIEVersion == 8) {
                        return 8;
                    } else if (fIEVersion == 9) {
                        return 9;
                    } else if (fIEVersion == 10) {
                        return 10;
                    } else {
                        return 6; // ie <=7
                    }
                } else if (isEdge) {
                    return 'edge'
                } else if (isIE11) {
                    return 11;
                } else {
                    return -1; // not ie
                }
            }
            global.sm4Encode = function (text) {
                var inputtext = text;
                var keytext = global.decodeKey;
                var inputBytes = Hex.utf8StrToBytes(inputtext);
                var key = Hex.decode(keytext);
                var sm4 = new SM4();
                var cipher = sm4.encrypt_ecb(key, inputBytes);
                if (!cipher) {
                    console.log('加密异常')
                    return
                }
                return Hex.encode(cipher, 0, cipher.length)
            }
            // 封装弹窗组件
            global.dialog = function(obj) {
                if (!obj.dialogId) {
                    console.log('dialog id is null');
                    return;
                }
                // dialogTimes++;
                var dialog = $('#' + obj.dialogId);
                // 事件处理
                if (obj.init != undefined) {
                    obj.init();
                }
                dialog.find(".ybj-dialog-confirm").unbind("click");
                if (obj.confirm != undefined) {
                    dialog.find(".ybj-dialog-confirm").click(obj.confirm);
                }
                dialog.find(".ybj-dialog-cancel").unbind("click");
                if (obj.cancel != undefined) {
                    dialog.find(".ybj-dialog-cancel").click(obj.cancel);
                } else {
                    dialog.hide();
                    $('body .dialog-mask').remove();
                }
                // 添加蒙层，弹窗显示
                // $('<div class="dialog-mask"></div>').appendTo($('body'));
                $('body').append($('<div class="dialog-mask"></div>'));
                $('body .dialog-mask').show();
                dialog.show();
                // close事件
                this.dialog.close = function() {
                    dialog.hide();
                    $('body .dialog-mask').remove();
                }
                return this.dialog;
            }
            return global;
        };
    });
});

app.directive("pager", function (global) {
    return {
        restrict: 'A',
        templateUrl: '/cn/view/components/pager.html',
        controller: function ($scope) {
            // if($scope.data !=undefined){
            // console.log("pageindex:"+global.getPageIndex());

            if ($scope.pager == undefined) {
                $scope.pager = {
                    itemNum: 0, // 数据总条数
                    pageItemNum: 10, // 每页条数
                    currentPageIndex: global.getPageIndex(), // 当前页码
                    displayPageNum: 3, // pager显示页数
                    pageIndexChange: function (currentPageIndex) { },
                    jumpInputenterval: ''
                };
            };
            $scope.pager.location = location.protocol + "//" + location.host + location.pathname;
            $scope.pager._init = function () {
                if ($scope.pager.itemNum == 0) {
                    $scope.pager.pageNum = 1;
                } else {
                    $scope.pager.pageNum = $scope.pager.itemNum % $scope.pager.pageItemNum == 0 ? Math.floor($scope.pager.itemNum / $scope.pager.pageItemNum) : Math.floor($scope.pager.itemNum / $scope.pager.pageItemNum) + 1;
                }

                $scope.pager._startPageIndex = $scope.pager.currentPageIndex % $scope.pager.displayPageNum == 0 ? Math.floor(($scope.pager.currentPageIndex - 1) / $scope.pager.displayPageNum) * $scope.pager.displayPageNum + 1 : Math.floor($scope.pager.currentPageIndex / $scope.pager.displayPageNum) * $scope.pager.displayPageNum + 1;

                $scope.pager._pageIndexArr = [];
                $scope.pager._pageIndexArr2 = [];
                for (var i = 0; i < $scope.pager.pageNum; i++) {
                    $scope.pager._pageIndexArr.push(i + 1);
                }

                for (var i = $scope.pager._startPageIndex, j = 1; i <= $scope.pager._pageIndexArr.length && j <= $scope.pager.displayPageNum; i++ , j++) {
                    $scope.pager._pageIndexArr2.push($scope.pager._pageIndexArr[i - 1]);
                }
                if ($scope.pager._pageIndexArr2[$scope.pager._pageIndexArr2.length - 1] < $scope.pager._pageIndexArr[$scope.pager._pageIndexArr.length - 1]) {
                    $scope.pager._morePage = true;
                } else {
                    $scope.pager._morePage = false;
                }
            };
            $scope.pager.init = $scope.pager._init;
            $scope.pager._init();

            $scope.pager.more = function () {
                //console.log("more");
                $scope.pager.currentPageIndex = $scope.pager._pageIndexArr2[$scope.pager._pageIndexArr2.length - 1] + 1;

                if ($scope.pager.pageIndexChange != undefined) {
                    $scope.pager.pageIndexChange($scope.pager.currentPageIndex);
                }
                $scope.pager._init();
            }
            $scope.pager.goTo = function (index) {
                //console.log("go to page ", index);
                $scope.pager.currentPageIndex = index;
                location.hash = "#" + index;
                if ($scope.pager.pageIndexChange != undefined) {
                    $scope.pager.pageIndexChange(index);
                }
                $scope.pager._init();
            }
            $scope.pager.next = function () {
                //console.log("next");
                if ($scope.pager.currentPageIndex + 1 <= $scope.pager.pageNum) {
                    $scope.pager.currentPageIndex++;
                }

                if ($scope.pager.pageIndexChange != undefined) {
                    $scope.pager.pageIndexChange($scope.pager.currentPageIndex);
                }
                $scope.pager._init();
            }
            $scope.pager.prev = function () {
                //console.log("prev ");
                if ($scope.pager.currentPageIndex - 1 >= 1) {
                    $scope.pager.currentPageIndex--;
                }

                if ($scope.pager.pageIndexChange != undefined) {
                    $scope.pager.pageIndexChange($scope.pager.currentPageIndex);
                }
                $scope.pager._init();
            }

            //console.log($scope);
        }
        //  }
    };
});
app.directive("pager4", function (global) {
  return {
      restrict: 'A',
      templateUrl: '/cn/view/components/pager4.html',
      controller: function ($scope) {
          // if($scope.data !=undefined){
          // console.log("pageindex:"+global.getPageIndex());
          if ($scope.pager == undefined) {
              $scope.pager = {
                  itemNum: 0, // 数据总条数
                  pageItemNum: 10, // 每页条数
                  currentPageIndex: global.getPageIndex2(), // 当前页码
                  displayPageNum: 3, // pager显示页数
                  pageIndexChange: function (currentPageIndex) { },
                  jumpInputenterval: ''
              };
          }

          $scope.pager.location = location.protocol + "//" + location.host + location.pathname;
          $scope.pager._init = function () {

              if ($scope.pager.itemNum == 0) {
                  $scope.pager.pageNum = 1;
              }
              else {
                  $scope.pager.pageNum = $scope.pager.itemNum % $scope.pager.pageItemNum == 0 ? Math.floor($scope.pager.itemNum / $scope.pager.pageItemNum) : Math.floor($scope.pager.itemNum / $scope.pager.pageItemNum) + 1;
              }
              $scope.pager._startPageIndex = $scope.pager.currentPageIndex % $scope.pager.displayPageNum == 0 ? Math.floor(($scope.pager.currentPageIndex - 1) / $scope.pager.displayPageNum) * $scope.pager.displayPageNum + 1 : Math.floor($scope.pager.currentPageIndex / $scope.pager.displayPageNum) * $scope.pager.displayPageNum + 1;

              $scope.pager._pageIndexArr = [];
              $scope.pager._pageIndexArr2 = [];
              for (var i = 0; i < $scope.pager.pageNum; i++) {
                  $scope.pager._pageIndexArr.push(i + 1);
              }

              for (var i = $scope.pager._startPageIndex, j = 1; i <= $scope.pager._pageIndexArr.length && j <= $scope.pager.displayPageNum; i++ , j++) {
                  $scope.pager._pageIndexArr2.push($scope.pager._pageIndexArr[i - 1]);
              }
              if ($scope.pager._pageIndexArr2[$scope.pager._pageIndexArr2.length - 1] < $scope.pager._pageIndexArr[$scope.pager._pageIndexArr.length - 1]) {
                  $scope.pager._morePage = true;
              } else {
                  $scope.pager._morePage = false;
              }
          };
          $scope.pager.init = $scope.pager._init;
          $scope.pager._init();
          $scope.refreshPager = function (total) {
              $scope.pager.itemNum = total;
              $scope.pager._init();
          }
          $scope.pager.pageIndexChange(global.getPageIndex2(), function (total) {
              $scope.refreshPager(total);
          });
          $scope.pager.more = function () {
              //console.log("more");
              $scope.pager.jumpInputenterval = "";
              $scope.pager.currentPageIndex = $scope.pager._pageIndexArr2[$scope.pager._pageIndexArr2.length - 1] + 1;
              location.hash = "#" + $scope.pager.currentPageIndex;
              if ($scope.pager.pageIndexChange != undefined) {
                  $scope.pager.pageIndexChange($scope.pager.currentPageIndex, function (total) {
                      $scope.refreshPager(total);
                  });
              }
              $scope.pager._init();
          }
          $scope.pager.goTo = function (index) {

              //console.log("go to page ", index);
              if (index <= 0) {
                  // $scope.toast("请输入页码范围内的值");
                  index = 1;

              }
              if (index > $scope.pager.pageNum) {
                  index = $scope.pager.pageNum;
              }
              $scope.pager.currentPageIndex = index;
              $scope.pager.jumpInputenterval = $scope.pager.currentPageIndex;
              location.hash = "#" + index;
              if ($scope.pager.pageIndexChange != undefined) {
                  $scope.pager.pageIndexChange(index, function (total) {
                      $scope.refreshPager(total);
                  });

              }
              $scope.pager._init();
          }
          // add
          

          $scope.pager.goPage = function (index) {
            if(!index) {
              return
            }
            $scope.pager.jumpInputenterval = Number($("#jumpInputText").val());
            var reg = /^\+?[1-9][0-9]*$/;
                if (!reg.test($scope.pager.jumpInputenterval)) {
                    return false;
                }
                if ($scope.pager.jumpInputenterval > $scope.pager.pageNum) {
                    return false;
                }
                $scope.pager.goTo($scope.pager.jumpInputenterval);
                $scope.pager.currentPageIndex = $scope.pager.jumpInputenterval;
                location.hash = "#" + $scope.pager.jumpInputenterval;
                $scope.pager._init();
        }
          $("#jumpInputText").keyup(function (e) {
              // console.log("jumpInputenterval");
              $scope.pager.jumpInputenterval = Number($("#jumpInputText").val());
              // console.log( $scope.pager.jumpInputenterval);
              var reg = /^\+?[1-9][0-9]*$/;

              if (e.keyCode == 13) {
                  // console.log('1');
                  if (!reg.test($scope.pager.jumpInputenterval)) {
                      // console.log($scope.pager.currentPageIndex);
                      // $scope.pager.jumpInputenterval = location.hash.substr(1)
                      return false;
                  }
                  if ($scope.pager.jumpInputenterval > $scope.pager.pageNum) {
                      return false;
                  }
                  // $scope.pager.currentPageIndex = $scope.pager.jumpInputenterval;
                  $scope.pager.goTo($scope.pager.jumpInputenterval);
                  $scope.pager.currentPageIndex = $scope.pager.jumpInputenterval;
                  location.hash = "#" + $scope.pager.jumpInputenterval;
                  // if ($scope.pager.pageIndexChange != undefined) {
                  //     $scope.pager.pageIndexChange($scope.pager.currentPageIndex, function (total) {
                  //         $scope.refreshPager(total);
                  //     });
                  // }
                  $scope.pager._init();

              }
          })

          // $scope.pager.jumpbuttonclick = function () {
          //     // var jumpInputclick=$("#jumpInputText").val();
          //     var jumpval = $("#jumpInputText").val();
          //     console.log(jumpval);
          //     if (event.keyCode == 13) {
          //         var jumpval = $("#jumpInputText").val();
          //         $scope.pager.goTo(jumpval);
          //     }
          //     $scope.pager.goTo(jumpval);
          //     console.log('jumpbuttonclick');
          //     $scope.pager.currentPageIndex = jumpval;
          //     location.hash = "#" + jumpval;

          //     if ($scope.pager.pageIndexChange != undefined) {
          //         $scope.pager.pageIndexChange($scope.pager.currentPageIndex, function (total) {
          //             $scope.refreshPager(total);
          //         });
          //     }
          //     $scope.pager._init();

          // }
          // add


          $scope.pager.next = function () {
              //console.log("next");
              $scope.pager.jumpInputenterval = "";
              if ($scope.pager.currentPageIndex - 0 + 1 <= $scope.pager.pageNum) {
                  $scope.pager.currentPageIndex++;
                  location.hash = "#" + $scope.pager.currentPageIndex;
              } else {
                  return false;
              }

              if ($scope.pager.pageIndexChange != undefined) {
                  $scope.pager.pageIndexChange($scope.pager.currentPageIndex, function (total) {
                      $scope.refreshPager(total);
                  });
              }
              $scope.pager._init();


          }
          $scope.pager.prev = function () {
              //console.log("prev ");
              $scope.pager.jumpInputenterval = "";
              if ($scope.pager.currentPageIndex - 1 >= 1) {
                  $scope.pager.currentPageIndex--;
                  location.hash = "#" + $scope.pager.currentPageIndex;
              } else {
                  return false;
              }

              if ($scope.pager.pageIndexChange != undefined) {
                  $scope.pager.pageIndexChange($scope.pager.currentPageIndex, function (total) {
                      $scope.refreshPager(total);
                  });
              }
              $scope.pager._init();
          }
          $scope.pager.first = function () {
              $scope.pager.jumpInputenterval = "";
              $scope.pager.currentPageIndex = 1;
              location.hash = "#" + $scope.pager.currentPageIndex;
              if ($scope.pager.pageIndexChange != undefined) {
                  $scope.pager.pageIndexChange($scope.pager.currentPageIndex, function (total) {
                      $scope.refreshPager(total);
                  });
              }
              $scope.pager._init();
          }
          $scope.pager.last = function () {
              $scope.pager.jumpInputenterval = "";
              $scope.pager.currentPageIndex = $scope.pager.pageNum;
              location.hash = "#" + $scope.pager.currentPageIndex;
              if ($scope.pager.pageIndexChange != undefined) {
                  $scope.pager.pageIndexChange($scope.pager.currentPageIndex, function (total) {
                      $scope.refreshPager(total);
                  });
              }
              $scope.pager._init();
          }
          // console.log('pagerloaded broadcast')
          $scope.$broadcast('pagerLoaded', {});

          //按z显示数据总条数，按x隐藏
          $(document).keydown(function (evt) {
              if (evt.keyCode == 90) {
                  $scope.displayitemNum = true;
                  $scope.$apply();
              } else if (evt.keyCode == 88) {
                  $scope.displayitemNum = false;
                  $scope.$apply();
              }
          })
      }
      //  }
  };
});
app.directive("pager2", function (global) {
    return {
        restrict: 'A',
        templateUrl: '/cn/view/components/pager2.html',
        controller: function ($scope) {
            // if($scope.data !=undefined){
            // console.log("pageindex:"+global.getPageIndex());
            if ($scope.pager == undefined) {
                $scope.pager = {
                    itemNum: 0, // 数据总条数
                    pageItemNum: 10, // 每页条数
                    currentPageIndex: global.getPageIndex2(), // 当前页码
                    displayPageNum: 3, // pager显示页数
                    pageIndexChange: function (currentPageIndex) { },
                    jumpInputenterval: ''
                };
            }

            $scope.pager.location = location.protocol + "//" + location.host + location.pathname;
            $scope.pager._init = function () {

                if ($scope.pager.itemNum == 0) {
                    $scope.pager.pageNum = 1;
                }
                else {
                    $scope.pager.pageNum = $scope.pager.itemNum % $scope.pager.pageItemNum == 0 ? Math.floor($scope.pager.itemNum / $scope.pager.pageItemNum) : Math.floor($scope.pager.itemNum / $scope.pager.pageItemNum) + 1;
                }
                $scope.pager._startPageIndex = $scope.pager.currentPageIndex % $scope.pager.displayPageNum == 0 ? Math.floor(($scope.pager.currentPageIndex - 1) / $scope.pager.displayPageNum) * $scope.pager.displayPageNum + 1 : Math.floor($scope.pager.currentPageIndex / $scope.pager.displayPageNum) * $scope.pager.displayPageNum + 1;

                $scope.pager._pageIndexArr = [];
                $scope.pager._pageIndexArr2 = [];
                for (var i = 0; i < $scope.pager.pageNum; i++) {
                    $scope.pager._pageIndexArr.push(i + 1);
                }

                for (var i = $scope.pager._startPageIndex, j = 1; i <= $scope.pager._pageIndexArr.length && j <= $scope.pager.displayPageNum; i++ , j++) {
                    $scope.pager._pageIndexArr2.push($scope.pager._pageIndexArr[i - 1]);
                }
                if ($scope.pager._pageIndexArr2[$scope.pager._pageIndexArr2.length - 1] < $scope.pager._pageIndexArr[$scope.pager._pageIndexArr.length - 1]) {
                    $scope.pager._morePage = true;
                } else {
                    $scope.pager._morePage = false;
                }
            };
            $scope.pager.init = $scope.pager._init;
            $scope.pager._init();
            $scope.refreshPager = function (total) {
                $scope.pager.itemNum = total;
                $scope.pager._init();
            }
            $scope.pager.pageIndexChange(global.getPageIndex2(), function (total) {
                $scope.refreshPager(total);
            });
            $scope.pager.more = function () {
                //console.log("more");
                $scope.pager.jumpInputenterval = "";
                $scope.pager.currentPageIndex = $scope.pager._pageIndexArr2[$scope.pager._pageIndexArr2.length - 1] + 1;
                location.hash = "#" + $scope.pager.currentPageIndex;
                if ($scope.pager.pageIndexChange != undefined) {
                    $scope.pager.pageIndexChange($scope.pager.currentPageIndex, function (total) {
                        $scope.refreshPager(total);
                    });
                }
                $scope.pager._init();
            }
            $scope.pager.goTo = function (index) {

                //console.log("go to page ", index);
                if (index <= 0) {
                    // $scope.toast("请输入页码范围内的值");
                    index = 1;

                }
                if (index > $scope.pager.pageNum) {
                    index = $scope.pager.pageNum;
                }
                $scope.pager.currentPageIndex = index;
                $scope.pager.jumpInputenterval = $scope.pager.currentPageIndex;
                location.hash = "#" + index;
                if ($scope.pager.pageIndexChange != undefined) {
                    $scope.pager.pageIndexChange(index, function (total) {
                        $scope.refreshPager(total);
                    });

                }
                $scope.pager._init();
            }
            // add


            $("#jumpInputText").keyup(function (e) {
                // console.log("jumpInputenterval");
                $scope.pager.jumpInputenterval = Number($("#jumpInputText").val());
                // console.log( $scope.pager.jumpInputenterval);
                var reg = /^\+?[1-9][0-9]*$/;

                if (e.keyCode == 13) {
                    // console.log('1');
                    if (!reg.test($scope.pager.jumpInputenterval)) {
                        // console.log($scope.pager.currentPageIndex);
                        // $scope.pager.jumpInputenterval = location.hash.substr(1)
                        return false;
                    }
                    if ($scope.pager.jumpInputenterval > $scope.pager.pageNum) {
                        return false;
                    }
                    // $scope.pager.currentPageIndex = $scope.pager.jumpInputenterval;
                    $scope.pager.goTo($scope.pager.jumpInputenterval);
                    $scope.pager.currentPageIndex = $scope.pager.jumpInputenterval;
                    location.hash = "#" + $scope.pager.jumpInputenterval;
                    // if ($scope.pager.pageIndexChange != undefined) {
                    //     $scope.pager.pageIndexChange($scope.pager.currentPageIndex, function (total) {
                    //         $scope.refreshPager(total);
                    //     });
                    // }
                    $scope.pager._init();

                }
            })

            // $scope.pager.jumpbuttonclick = function () {
            //     // var jumpInputclick=$("#jumpInputText").val();
            //     var jumpval = $("#jumpInputText").val();
            //     console.log(jumpval);
            //     if (event.keyCode == 13) {
            //         var jumpval = $("#jumpInputText").val();
            //         $scope.pager.goTo(jumpval);
            //     }
            //     $scope.pager.goTo(jumpval);
            //     console.log('jumpbuttonclick');
            //     $scope.pager.currentPageIndex = jumpval;
            //     location.hash = "#" + jumpval;

            //     if ($scope.pager.pageIndexChange != undefined) {
            //         $scope.pager.pageIndexChange($scope.pager.currentPageIndex, function (total) {
            //             $scope.refreshPager(total);
            //         });
            //     }
            //     $scope.pager._init();

            // }
            // add


            $scope.pager.next = function () {
                //console.log("next");
                $scope.pager.jumpInputenterval = "";
                if ($scope.pager.currentPageIndex - 0 + 1 <= $scope.pager.pageNum) {
                    $scope.pager.currentPageIndex++;
                    location.hash = "#" + $scope.pager.currentPageIndex;
                } else {
                    return false;
                }

                if ($scope.pager.pageIndexChange != undefined) {
                    $scope.pager.pageIndexChange($scope.pager.currentPageIndex, function (total) {
                        $scope.refreshPager(total);
                    });
                }
                $scope.pager._init();


            }
            $scope.pager.prev = function () {
                //console.log("prev ");
                $scope.pager.jumpInputenterval = "";
                if ($scope.pager.currentPageIndex - 1 >= 1) {
                    $scope.pager.currentPageIndex--;
                    location.hash = "#" + $scope.pager.currentPageIndex;
                } else {
                    return false;
                }

                if ($scope.pager.pageIndexChange != undefined) {
                    $scope.pager.pageIndexChange($scope.pager.currentPageIndex, function (total) {
                        $scope.refreshPager(total);
                    });
                }
                $scope.pager._init();
            }
            $scope.pager.first = function () {
                $scope.pager.jumpInputenterval = "";
                $scope.pager.currentPageIndex = 1;
                location.hash = "#" + $scope.pager.currentPageIndex;
                if ($scope.pager.pageIndexChange != undefined) {
                    $scope.pager.pageIndexChange($scope.pager.currentPageIndex, function (total) {
                        $scope.refreshPager(total);
                    });
                }
                $scope.pager._init();
            }
            $scope.pager.last = function () {
                $scope.pager.jumpInputenterval = "";
                $scope.pager.currentPageIndex = $scope.pager.pageNum;
                location.hash = "#" + $scope.pager.currentPageIndex;
                if ($scope.pager.pageIndexChange != undefined) {
                    $scope.pager.pageIndexChange($scope.pager.currentPageIndex, function (total) {
                        $scope.refreshPager(total);
                    });
                }
                $scope.pager._init();
            }
            // console.log('pagerloaded broadcast')
            $scope.$broadcast('pagerLoaded', {});

            //按z显示数据总条数，按x隐藏
            $(document).keydown(function (evt) {
                if (evt.keyCode == 90) {
                    $scope.displayitemNum = true;
                    $scope.$apply();
                } else if (evt.keyCode == 88) {
                    $scope.displayitemNum = false;
                    $scope.$apply();
                }
            })
        }
        //  }
    };
});
app.directive("pager3", function (global) {
    return {
        restrict: 'A',
        templateUrl: '/cn/view/components/pager3.html',
        controller: function ($scope) {
            // if($scope.data !=undefined){
            // console.log("pageindex:"+global.getPageIndex());
            if ($scope.pager == undefined) {
                $scope.pager = {
                    itemNum: 0, // 数据总条数
                    pageItemNum: 10, // 每页条数
                    currentPageIndex: global.getPageIndex2(), // 当前页码
                    displayPageNum: 3, // pager显示页数
                    pageIndexChange: function (currentPageIndex) { },
                    jumpInputenterval: ''
                };
            }

            $scope.pager.location = location.protocol + "//" + location.host + location.pathname;
            $scope.pager._init = function () {

                if ($scope.pager.itemNum == 0) {
                    $scope.pager.pageNum = 1;
                }
                else {
                    $scope.pager.pageNum = $scope.pager.itemNum % $scope.pager.pageItemNum == 0 ? Math.floor($scope.pager.itemNum / $scope.pager.pageItemNum) : Math.floor($scope.pager.itemNum / $scope.pager.pageItemNum) + 1;
                }
                $scope.pager._startPageIndex = $scope.pager.currentPageIndex % $scope.pager.displayPageNum == 0 ? Math.floor(($scope.pager.currentPageIndex - 1) / $scope.pager.displayPageNum) * $scope.pager.displayPageNum + 1 : Math.floor($scope.pager.currentPageIndex / $scope.pager.displayPageNum) * $scope.pager.displayPageNum + 1;

                $scope.pager._pageIndexArr = [];
                $scope.pager._pageIndexArr2 = [];
                for (var i = 0; i < $scope.pager.pageNum; i++) {
                    $scope.pager._pageIndexArr.push(i + 1);
                }

                for (var i = $scope.pager._startPageIndex, j = 1; i <= $scope.pager._pageIndexArr.length && j <= $scope.pager.displayPageNum; i++ , j++) {
                    $scope.pager._pageIndexArr2.push($scope.pager._pageIndexArr[i - 1]);
                }
                if ($scope.pager._pageIndexArr2[$scope.pager._pageIndexArr2.length - 1] < $scope.pager._pageIndexArr[$scope.pager._pageIndexArr.length - 1]) {
                    $scope.pager._morePage = true;
                } else {
                    $scope.pager._morePage = false;
                }
            };
            $scope.pager.init = $scope.pager._init;
            // $scope.pager._init();
            $scope.refreshPager = function (total) {
                $scope.pager.itemNum = total;
                $scope.pager._init();
            }
            // $scope.pager.pageIndexChange(global.getPageIndex2(), function (total) {
            //     $scope.refreshPager(total);
            // });
            $scope.pager.more = function () {
                //console.log("more");
                $scope.pager.jumpInputenterval = "";
                $scope.pager.currentPageIndex = $scope.pager._pageIndexArr2[$scope.pager._pageIndexArr2.length - 1] + 1;
                location.hash = "#" + $scope.pager.currentPageIndex;
                if ($scope.pager.pageIndexChange != undefined) {
                    $scope.pager.pageIndexChange($scope.pager.currentPageIndex, function (total) {
                        $scope.refreshPager(total);
                    });
                }
                $scope.pager._init();
            }
            $scope.pager.goTo = function (index) {

                //console.log("go to page ", index);
                if (index <= 0) {
                    // $scope.toast("请输入页码范围内的值");
                    index = 1;

                }
                if (index > $scope.pager.pageNum) {
                    index = $scope.pager.pageNum;
                }
                $scope.pager.currentPageIndex = index;
                $scope.pager.jumpInputenterval = $scope.pager.currentPageIndex;
                location.hash = "#" + index;
                if ($scope.pager.pageIndexChange != undefined) {
                    $scope.pager.pageIndexChange(index, function (total) {
                        $scope.refreshPager(total);
                    });

                }
                // $scope.pager._init();


            }
            // add


            $scope.myKeyup = function (e) {
                var keycode = window.event ? e.keyCode : e.which;
                if (keycode == 13) {
                    // debugger;
                    $scope.pager.jumpInputenterval;
                    var reg = /^\+?[1-9][0-9]*$/;


                    if (!reg.test($scope.pager.jumpInputenterval)) {
                        // console.log($scope.pager.currentPageIndex);
                        // $scope.pager.jumpInputenterval = location.hash.substr(1)
                        return false;
                    }
                    if ($scope.pager.jumpInputenterval > $scope.pager.pageNum) {
                        // console.log($scope.pager.currentPageIndex);
                        // location.hash=
                        // // $scope.pager.jumpInputenterval = location.hash.substr(1)
                        return false;
                    }
                    // $scope.pager.currentPageIndex = $scope.pager.jumpInputenterval;
                    $scope.pager.goTo($scope.pager.jumpInputenterval);
                    $scope.pager.currentPageIndex = $scope.pager.jumpInputenterval;
                    location.hash = "#" + $scope.pager.jumpInputenterval;
                    // if ($scope.pager.pageIndexChange != undefined) {
                    //     $scope.pager.pageIndexChange($scope.pager.currentPageIndex, function (total) {
                    //         $scope.refreshPager(total);
                    //     });
                    // }
                    $scope.pager._init();

                }
            }

            // $scope.pager.jumpbuttonclick = function () {
            //     // var jumpInputclick=$("#jumpInputText").val();
            //     var jumpval = $("#jumpInputText").val();
            //     console.log(jumpval);
            //     if (event.keyCode == 13) {
            //         var jumpval = $("#jumpInputText").val();
            //         $scope.pager.goTo(jumpval);
            //     }
            //     $scope.pager.goTo(jumpval);
            //     console.log('jumpbuttonclick');
            //     $scope.pager.currentPageIndex = jumpval;
            //     location.hash = "#" + jumpval;

            //     if ($scope.pager.pageIndexChange != undefined) {
            //         $scope.pager.pageIndexChange($scope.pager.currentPageIndex, function (total) {
            //             $scope.refreshPager(total);
            //         });
            //     }
            //     $scope.pager._init();

            // }
            // add


            $scope.pager.next = function () {
                //console.log("next");
                $scope.pager.jumpInputenterval = "";
                if ($scope.pager.currentPageIndex - 0 + 1 <= $scope.pager.pageNum) {
                    $scope.pager.currentPageIndex++;
                    location.hash = "#" + $scope.pager.currentPageIndex;
                } else {
                    return false;
                }

                if ($scope.pager.pageIndexChange != undefined) {
                    $scope.pager.pageIndexChange($scope.pager.currentPageIndex, function (total) {
                        $scope.refreshPager(total);
                    });
                }
                $scope.pager._init();


            }
            $scope.pager.prev = function () {
                //console.log("prev ");
                $scope.pager.jumpInputenterval = "";
                if ($scope.pager.currentPageIndex - 1 >= 1) {
                    $scope.pager.currentPageIndex--;
                    location.hash = "#" + $scope.pager.currentPageIndex;
                } else {
                    return false;
                }

                if ($scope.pager.pageIndexChange != undefined) {
                    $scope.pager.pageIndexChange($scope.pager.currentPageIndex, function (total) {
                        $scope.refreshPager(total);
                    });
                }
                $scope.pager._init();
            }
            $scope.pager.first = function () {
                $scope.pager.jumpInputenterval = "";
                $scope.pager.currentPageIndex = 1;
                location.hash = "#" + $scope.pager.currentPageIndex;
                if ($scope.pager.pageIndexChange != undefined) {
                    $scope.pager.pageIndexChange($scope.pager.currentPageIndex, function (total) {
                        $scope.refreshPager(total);
                    });
                }
                $scope.pager._init();
            }
            $scope.pager.last = function () {
                $scope.pager.jumpInputenterval = "";
                $scope.pager.currentPageIndex = $scope.pager.pageNum;
                location.hash = "#" + $scope.pager.currentPageIndex;
                if ($scope.pager.pageIndexChange != undefined) {
                    $scope.pager.pageIndexChange($scope.pager.currentPageIndex, function (total) {
                        $scope.refreshPager(total);
                    });
                }
                $scope.pager._init();
            }
            // console.log('pagerloaded broadcast')
            $scope.$broadcast('pagerLoaded', {});

            //按z显示数据总条数，按x隐藏
            $(document).keydown(function (evt) {
                if (evt.keyCode == 90) {
                    $scope.displayitemNum = true;
                    $scope.$apply();
                } else if (evt.keyCode == 88) {
                    $scope.displayitemNum = false;
                    $scope.$apply();
                }
            })
        }
        //  }
    };
});
app.directive("repeatFinish", function ($timeout) {
    return {
        restrict: 'A',
        link: function (scope, elem, attr) {
            if (scope.$last === true) {
                $timeout(function () {
                    scope.$emit('repeatFinishCallback');
                }, 100);
            }
        }
    }
});
app.filter("dateFormat", function () {
    return function (originalDate) {
        if (originalDate != undefined) {
            var d = new Date(Date.parse(originalDate.split(' ')[0].replace(/-/g, "/")));
            var month = d.getMonth() + 1 < 10 ? '0' + (d.getMonth() + 1) : d.getMonth() + 1;
            var date = d.getDate() < 10 ? '0' + d.getDate() : d.getDate();
            return month + '-' + date;
        } else {
            //console.log('ddd1')
            return "";
        }

    }
});
app.filter("dateFormat2", function () {
    return function (originalDate) {
        if (originalDate != undefined) {
            var d = new Date(Date.parse(originalDate.split(' ')[0].replace(/-/g, "/")));
            var month = d.getMonth() + 1 < 10 ? '0' + (d.getMonth() + 1) : d.getMonth() + 1;
            var date = d.getDate() < 10 ? '0' + d.getDate() : d.getDate();
            return d.getFullYear() + '-' + month + '-' + date;
        } else {
            //console.log('ddd2')
            return "";
        }

    }
});
app.filter("dateFormat3", function () {
    return function (originalDate) {
        if (originalDate != undefined) {
            return originalDate.substr(0, 10);
        } else {
            ///console.log('ddd3')
            return "";
        }

    }
});

app.filter("splitTxt60", function () {
  return function (txt) {
    if (!txt) {
      return;
  };
  if (txt.length < 60) {
      return str;
  }
  return str.substr(0, 60) + '...';
}
});

app.filter("splitTxt85", function () {
  return function (txt) {
    if (!txt) {
      return;
  };
  if (txt.length < 85) {
      return str;
  }
  return str.substr(0, 85) + '...';
}
});
// 政府行政法规等多链接中提取[]中的文本过滤器
app.filter("textFormat", function () {
    return function (urlARRchild) {
        var textrel = urlARRchild.split(/\[(.+?)\]/)[1];
        return textrel;

    }
});
// 政府行政法规等多链接中提取（）中的地址过滤器
app.filter("urlFormat", function () {
    return function (urlARRchild) {
        // var hrefrel=urlARRchild.split(/(\(|（)(.+?)(\)|）)/)[1];
        var hrefrel = urlARRchild.split(/(\(|（)(.+?)(\)|）)/)[2];
        return hrefrel;

    }
});
//为了繁体支持·
app.filter("ellipsis", ["$sce", function ($sce) {
    return function (originalStr, size) {
        if (originalStr == undefined || originalStr == '') {
            return '';
        } else {
            if (size >= originalStr.length) {
                return $sce.trustAsHtml(originalStr.substr(0, size));
            } else {
                var str = $sce.trustAsHtml(originalStr.substr(0, size) + '...');
                return $sce.trustAsHtml(originalStr.substr(0, size) + '...');

            }
        }
    }
}]);
app.filter("trustHtml", ["$sce", function ($sce) {
    return function (data) {
        return $sce.trustAsHtml(data);
    }
}]);

app.filter("trustAsResourceUrl", ["$sce", function ($sce) {
    return function (data) {
        return $sce.trustAsResourceUrl(data);
    }
}]);
// // 去除html标签
// app.filter("trimHtml", [function ($sce) {
//     return function (data) {
//         if (data != undefined) {
//             // var aftertrimHtml = data.replace(/<[^>]+>/g, '');  //去掉html标签
//             // return aftertrimHtml.replace(/&nbsp;/ig, '');   //去掉&nbsp
//              // return data.replace(/<a.*?>(.*?)<\/a>/ig, '');
//              var RegExp=/<a.*?>(.*?)<\/a>/ig;
//              var result;
//              if((result=RegExp.exec(data))!=null){
//                  var  trimAtag= result[1]
//                  var aftertrimAtag = trimAtag.replace(/<[^>]+>/g, '');  //去掉html标签
//                  return aftertrimAtag.replace(/&nbsp;/ig, '');   //去掉&nbsp
//              }
//              else{
//                  var aftertrimHtml = data.replace(/<[^>]+>/g, '');  //去掉html标签
//                  return aftertrimHtml.replace(/&nbsp;/ig, '');   //去掉&nbsp
//              }
//         } else {
//             return false;
//         }
//     }
// }]);
// 去除html标签
app.filter("trimHtml", [function ($sce) {
    return function (data) {
        if (data != undefined) {
            var RegExp = /<a.*?>(.*?)<\/a>/ig;
            var result;
            var RegExp2 = /<font color="red">/ig;
            var RegExp3 = /<\/font>/ig;
            if ((result = RegExp.exec(data)) != null) {
                if (RegExp2.test(data)) {
                    return  result[1].replace(RegExp2, '').replace(RegExp3, '');
                }else{
                    return result[1];
                }  }
            else {
                if (RegExp2.test(data)) {
                    return  data.replace(RegExp2, '').replace(RegExp3, '');
                }else{
                    return data;
                } }
        } else {
            return false;
        }
    }
}]);
//去除html标签
// app.filter("trimHtml", [function ($sce) {
//     return function (data) {
//         if (data != undefined) {
//             // var aftertrimHtml = data.replace(/<[^>]+>/g, '');  //去掉html标签
//             // return aftertrimHtml.replace(/&nbsp;/ig, '');   //去掉&nbsp
//             // return data.replace(/<a.*?>(.*?)<\/a>/ig, '');
//             var RegExp = /<a.*?>(.*?)<\/a>/ig;
//             var result;
//             var result2;
//             if ((result = RegExp.exec(data)) != null) {
//                 var trimAtag = result[1]
//                 // var RegExp2 = /<font.*?>(\\s\\S*?)<\/font>/ig;
//                 var RegExp2 = /<\/?font.*?>/ig;
//                 if ((result2 = RegExp2.exec(trimAtag)) != null) {
//                     return result2[1];
//                 }
//                 else {
//                     return result[1];
//                 }
//                 //  var  trimAtag= result[1];
//                 //  var aftertrimAtag = trimAtag.replace(/<[^>]+>/g, '');  //去掉html标签
//                 //  return aftertrimAtag.replace(/&nbsp;/ig, '');   //去掉&nbsp
//             }
//             else {
//                 return data;
//             }
//         } else {
//             return false;
//         }
//     }
// }]);
//去除html标签政府信息公开检索页面专用
app.filter("trimHtml2", [function ($sce) {
    return function (data) {
        if (data != undefined) {
            var aftertrimHtml = data.replace(/<[^>]+>/g, '');  //去掉html标签
            return aftertrimHtml.replace(/&nbsp;/ig, '');   //去掉&nbsp
        } else {
            return false;
        }

    }
}]);


//判断是否含有a标签
app.filter("isAHtml", [function ($sce) {
    return function (data) {
        if (data != undefined) {
            return data.toLowerCase().indexOf('a>') > -1;
        } else {
            return false;
        }

    }
}]);


//判断itemUrl是否含有http，表示跳转到http地址
app.filter("ishttpUrl", [function ($sce) {
    return function (data) {
        if (data != undefined) {
            return data.toLowerCase().substring(0, 4).indexOf('http') > -1;
        } else {
            return false;
        }

    }
}]);

//判断itemUrl是否含有relative，表示跳转到相对地址
app.filter("isrelativeUrl", [function ($sce) {
    return function (data) {
        if (data != undefined) {
            return data.toLowerCase().substring(0, 8).indexOf('relative') > -1;
        } else {
            return false;
        }
    }
}]);

//判断itemUrl是否含有noclick，表示不能点击
app.filter("isnoclickUrl", [function ($sce) {
    return function (data) {
        if (data != undefined) {
            return data.toLowerCase().substring(0, 7).indexOf('noclick') > -1;
        } else {
            return false;
        }
    }
}]);

//面包屑组装url
app.filter("breadcrumbitemUrl", [function ($sce) {
    var prefix = "/cn/view/pages/";
    return function (data) {
        if (data != undefined) {
            if (data.itemUrl.toLowerCase().substring(0, 8).indexOf('relative') > -1) {
                return data.itemUrl.substr(9);
            } else if (data.itemUrl.toLowerCase().substring(0, 9).indexOf('tosubmenu') > -1) {
                return prefix + 'ItemList.html?itemPId=' + data.itemPid + '&itemId=' + data.subItemslist[0].itemId + '&itemUrl=' + data.subItemslist[0].itemUrl + '&itemName=' + data.subItemslist[0].itemName;
            } else if (data.itemName.indexOf("图片新闻") > -1 || data.itemName.indexOf("机构监管") > -1 || data.itemName.indexOf("功能监管") > -1 || data.itemName.indexOf("综合管理") > -1) {
                return "";
            } else if (data.itemUrl.indexOf('ItemListRightList') > -1) {
                return prefix + 'ItemList.html?itemPId=' + data.itemPPid + '&itemId=' + data.itemId + '&itemUrl=' + data.itemUrl + '&itemName=' + data.itemName + '&itemsubPId=' + data.itemsubPId;
            } else if (data.itemUrl.indexOf('ItemListRightArticle') > -1) {
                return prefix + 'ItemList.html?itemPId=' + data.itemPid + '&itemId=' + data.itemId + '&itemUrl=' + data.itemUrl + '&itemName=' + data.itemName;
            } else if (data.itemUrl == null || data.itemUrl == undefined || data.itemUrl == "" || data.itemUrl.indexOf('noclick') > -1) {
                return "";
            } else {
                return prefix + 'ItemList.html?itemPId=' + data.itemPid + '&itemId=' + data.itemId + '&itemUrl=' + data.itemUrl + '&itemName=' + data.itemName;
            }
        } else {
            return false;
        }
    }
}]);

// 栏目itemUrl过滤
app.filter("parseItemUrl", [function ($sce) {
    return function (itemUrl, pid, id) {
        var prefix = "/cn/view/pages/"
        if (itemUrl != undefined) {
            // 相对路径
            if (itemUrl.toLowerCase().substring(0, 8).indexOf('relative') > -1) {
                return prefix + itemUrl.toLowerCase().replace('relative:', '');
            }
            // 不能点击 
            else if (itemUrl.toLowerCase().substring(0, 7).indexOf('noclick') > -1) {
                return '';
            }
            // 绝对路径 
            else if (itemUrl.toLowerCase().substring(0, 4).indexOf('http') > -1) {
                return itemUrl;
            }
            // itemList.html
            else {
                if (pid != undefined && id != undefined) {
                    return prefix + 'ItemList.html?itemPId=' + pid + '&itemId=' + id + '&itemUrl=' + itemUrl;
                } else {
                    return '';
                }
            }
        } else {
            // console.log('yyy');
            return '';
        }
    }
}]);
// 栏目itemUrl过滤
app.filter("parseItemUrl2", [function ($sce) {
    return function (item, pid) {
        var prefix = "/cn/view/pages/"
        if (item != undefined) {
            if (item.itemUrl != undefined) {
                // 相对路径
                if (item.itemUrl.toLowerCase().substring(0, 8).indexOf('relative') > -1) {
                    return prefix + item.itemUrl.toLowerCase().replace('relative:', '');
                }
                // 不能点击 
                else if (item.itemUrl.toLowerCase().substring(0, 7).indexOf('noclick') > -1) {
                    return '';
                }
                // 绝对路径 
                else if (item.itemUrl.toLowerCase().substring(0, 4).indexOf('http') > -1) {
                    return item.itemUrl;
                } else {
                    if (pid == undefined) {
                        return prefix + 'ItemList.html?itemPId=' + item.itemPid + '&itemId=' + item.itemId + '&itemUrl=' + item.itemUrl;
                    } else {
                        return prefix + 'ItemList.html?itemPId=' + pid + '&itemId=' + item.itemId + '&itemUrl=' + item.itemUrl;
                    }
                }
            }

        } else {
            // console.log('xxx')
            return '';
        }
    }
}]);

//遇到\n替换成<br>
app.filter("addBr", ["$sce", function ($sce) {
    return function (data) {
        if (data != null) {
            return $sce.trustAsHtml(data.replace(/\n/g, "<br>"));
        }
    }
}]);
//遇到文号中的[]替换成〔〕〕〕﹝﹞〔〕
app.filter("hexagonSymbol", ["$sce", function ($sce) {
    return function (data) {
        if (data != null) {
            return $sce.trustAsHtml(data.replace(/\[/g, "〔").replace(/\]/g, "〕"));
        }
    }
}]);


