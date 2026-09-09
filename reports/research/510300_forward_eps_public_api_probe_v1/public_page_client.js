/******/ (function(modules) { // webpackBootstrap
/******/ 	// The module cache
/******/ 	var installedModules = {};
/******/
/******/ 	// The require function
/******/ 	function __webpack_require__(moduleId) {
/******/
/******/ 		// Check if module is in cache
/******/ 		if(installedModules[moduleId]) {
/******/ 			return installedModules[moduleId].exports;
/******/ 		}
/******/ 		// Create a new module (and put it into the cache)
/******/ 		var module = installedModules[moduleId] = {
/******/ 			i: moduleId,
/******/ 			l: false,
/******/ 			exports: {}
/******/ 		};
/******/
/******/ 		// Execute the module function
/******/ 		modules[moduleId].call(module.exports, module, module.exports, __webpack_require__);
/******/
/******/ 		// Flag the module as loaded
/******/ 		module.l = true;
/******/
/******/ 		// Return the exports of the module
/******/ 		return module.exports;
/******/ 	}
/******/
/******/
/******/ 	// expose the modules object (__webpack_modules__)
/******/ 	__webpack_require__.m = modules;
/******/
/******/ 	// expose the module cache
/******/ 	__webpack_require__.c = installedModules;
/******/
/******/ 	// define getter function for harmony exports
/******/ 	__webpack_require__.d = function(exports, name, getter) {
/******/ 		if(!__webpack_require__.o(exports, name)) {
/******/ 			Object.defineProperty(exports, name, { enumerable: true, get: getter });
/******/ 		}
/******/ 	};
/******/
/******/ 	// define __esModule on exports
/******/ 	__webpack_require__.r = function(exports) {
/******/ 		if(typeof Symbol !== 'undefined' && Symbol.toStringTag) {
/******/ 			Object.defineProperty(exports, Symbol.toStringTag, { value: 'Module' });
/******/ 		}
/******/ 		Object.defineProperty(exports, '__esModule', { value: true });
/******/ 	};
/******/
/******/ 	// create a fake namespace object
/******/ 	// mode & 1: value is a module id, require it
/******/ 	// mode & 2: merge all properties of value into the ns
/******/ 	// mode & 4: return value when already ns object
/******/ 	// mode & 8|1: behave like require
/******/ 	__webpack_require__.t = function(value, mode) {
/******/ 		if(mode & 1) value = __webpack_require__(value);
/******/ 		if(mode & 8) return value;
/******/ 		if((mode & 4) && typeof value === 'object' && value && value.__esModule) return value;
/******/ 		var ns = Object.create(null);
/******/ 		__webpack_require__.r(ns);
/******/ 		Object.defineProperty(ns, 'default', { enumerable: true, value: value });
/******/ 		if(mode & 2 && typeof value != 'string') for(var key in value) __webpack_require__.d(ns, key, function(key) { return value[key]; }.bind(null, key));
/******/ 		return ns;
/******/ 	};
/******/
/******/ 	// getDefaultExport function for compatibility with non-harmony modules
/******/ 	__webpack_require__.n = function(module) {
/******/ 		var getter = module && module.__esModule ?
/******/ 			function getDefault() { return module['default']; } :
/******/ 			function getModuleExports() { return module; };
/******/ 		__webpack_require__.d(getter, 'a', getter);
/******/ 		return getter;
/******/ 	};
/******/
/******/ 	// Object.prototype.hasOwnProperty.call
/******/ 	__webpack_require__.o = function(object, property) { return Object.prototype.hasOwnProperty.call(object, property); };
/******/
/******/ 	// __webpack_public_path__
/******/ 	__webpack_require__.p = "";
/******/
/******/
/******/ 	// Load entry module and return exports
/******/ 	return __webpack_require__(__webpack_require__.s = 390);
/******/ })
/************************************************************************/
/******/ ({

/***/ 0:
/***/ (function(module, exports) {

module.exports = moment;

/***/ }),

/***/ 1:
/***/ (function(module, exports, __webpack_require__) {

"use strict";

/**
 * 数据字段常用公用方法
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
exports.__esModule = true;
var moment_1 = __importDefault(__webpack_require__(0));
var weeks = ['日', '一', '二', '三', '四', '五', '六'];
var quarters = ['1', '1', '1', '2', '2', '2', '3', '3', '3', '4', '4', '4'];
var quarters_z = ['一', '一', '一', '二', '二', '二', '三', '三', '三', '四', '四', '四'];
var quarters0 = ['1', '1', '1', '1-2', '1-2', '1-2', '1-3', '1-3', '1-3', '1-4', '1-4', '1-4'];
var quarters0_z = ['一', '一', '一', '一-二', '一-二', '一-二', '一-三', '一-三', '一-三', '一-四', '一-四', '一-四'];
var reports = ['1季', '1季', '1季', '中', '中', '中', '3季', '3季', '3季', '年', '年', '年'];
var reports_z = ['一季', '一季', '一季', '中', '中', '中', '三季', '三季', '三季', '年', '年', '年'];
var defaultfmtconfig = {
    def: '-',
    omit: true,
    trim: undefined,
    strfmt: undefined,
    num: 432,
    smart: false,
    fix: '',
    zoom: 0,
    showcolor: false,
    showabs: false,
    qfw: false,
    colorfmt: '<span class="{color}">{value}</span>',
    gtcolor: 'red',
    ltcolor: 'green',
    colorzero: 0
};
exports["default"] = {
    /**
     * 数据字段展示格式化方法(支持数字和日期处理)
     *@param {string} value:字段,一般为字符串或数字
     *@param {string} config:配置数据
     *@param {fmtconfig} config {
         
   ```
            def?: string;// 缺省值默认 ‘-’ 优先级最高，不受其他参数影响
            datefmt?: string;//日期展示格式，如果有值的话做日期处理，有值的话除def外其他字段全部作废 优先级小于def (详见日期处理方法dateformat)
            strfmt?: any;//作为字符串处理的参数，如果有值的话做字符串处理，优先级小于def,datefmt(详见字符串处理方法strformat)
            omit?: boolean;//是否自动改为省略格式【xxxx万|xxxx亿|xxxx万亿】默认true
            trim?: boolean;//是否自动省略末尾的0，默认不省略,smart有值时候则默认为true，可单独配置
            num?: any;//保留小数位，默认值为 432（3位数保留一位小数，2位和1位数保留两位小数），可以设置保留小数规则（"5421" 代表
                五位数字及以上保存0位小数，
                四位数字及以上保留一位小数，
                二位数字及以上保留2位小数，
                一位数字保留3位小数）
            smart?: number;//是否自动精简结果【最多保留多少位有效数字】配合omit=true时使用 有值的话小数位数小于num，>1时生效
            fix?: string;//显示单位 默认空   举例 【‘%’，‘只’】
            zoom?: number;//缩放倍数（指数幂） 默认0【不放大+】  举例 【1就是放大10倍】
            showabs?: boolean//是否展示为绝对值 默认不展示
            showcolor?: boolean//是否展示颜色 默认不展示
            colorfmt?: string;//带颜色的返回模板 默认<span class="{color}">{value}</span>
            gtcolor?: string;//大于标准值的样式 默认red
            ltcolor?: string;//小于标准值的样式 默认green
            colorvalue?: number;//颜色比较使用的值，默认使用value字段
            colorzero?: number;//颜色比较标准值 默认0  与零比较
            
   ```
        }
     **/
    fieldfmt: function (value, config) {
        if (this.isEmpty(value)) {
            return (config && config.def != undefined) ? config.def : defaultfmtconfig.def;
        }
        //无配置时尝试按数据类型转换
        if ($.isEmptyObject(config)) {
            if (/^\d{2,4}(\-|\/|\.)\d{1,2}\1\d{1,2}|^\d{13}$/.test(value)) { // 有bug
                return this.dateformat(value, 'YYYY-MM-DD');
            }
            if (!$.isNumeric(value)) {
                return value;
            }
        }
        config = $.extend({}, defaultfmtconfig, config);
        if (config && config.datefmt) {
            return this.dateformat(value, config.datefmt, config.def);
        }
        if (config && config.strfmt) {
            return this.strformat(value, config.strfmt, config.def);
        }
        if (isNaN(value) || !isFinite(value)) {
            return config.def;
        }
        var oldvalue = parseFloat(value);
        var fix = "";
        var num = 2;
        //@ts-ignore
        value = oldvalue * Math.pow(10, config.zoom).toFixed(Math.abs(config.zoom));
        if (config.showabs)
            value = Math.abs(value);
        var f = config.trim ? parseFloat : function (v) { return v; };
        var abslength = Math.abs(parseInt(value)).toString().length;
        if (config.omit) {
            if (abslength > 12) {
                value = value / 1000000000000;
                abslength -= 12;
                fix = '万亿';
            }
            else if (abslength > 8) {
                value = value / 100000000;
                abslength -= 8;
                fix = '亿';
            }
            else if (abslength > 4) {
                value = value / 10000;
                abslength -= 4;
                fix = '万';
            }
        }
        if (-1 < value && value < 1)
            abslength = 0;
        if (config.smart) {
            f = parseFloat;
            config.num = "43220";
        }
        if ($.isNumeric(config.num) && config.num < 10)
            num = config.num;
        else {
            var numstr = config.num.toString();
            num = numstr.length - 1;
            for (var i = 0; i < numstr.length; i++) {
                if (abslength >= numstr[i]) {
                    num = i;
                    break;
                }
            }
        }
        value = f(this.toFixed(value, num));
        value += fix + config.fix;
        if (config.showcolor) {
            var bz = config.colorvalue;
            if (!$.isNumeric(bz))
                bz = oldvalue;
            var color = bz == config.colorzero ? '' : bz > config.colorzero ? config.gtcolor : config.ltcolor;
            //修正数字过小造成的正负号丢失问题
            if (oldvalue < 0 && config.showabs == false && value[0] != '-')
                value = '-' + value;
            value = config.colorfmt.replace('{color}', color).replace('{value}', value);
        }
        if (config.qfw)
            return value.replace(/\d+/, function (n) {
                return n.replace(/(\d)(?=(\d{3})+$)/g, function ($1) {
                    return $1 + ",";
                });
            });
        return value;
    },
    /**
     * 日期字段展示格式化方法
     * @param {string} str:字段值+
     * @param {string} fmt：日期格式 根据nomentjs 文档http://momentjs.cn/docs/
     *                  添加 {W:"星期，['日', '一', '二', '三', '四', '五', '六']
     *                  添加 {Q|q:"季度，如3月为'一|1'；6月为'二|2'"；9月为'三|3'"；12月为'四|4'"}
     *                  添加 {J|j:"季度跨度，如3月为'一|1'；6月为'一-二|1-2'"；9月为'一-三|1-3'"；12月为示'一-四|1-4'"}
     *                  添加 {B|b:"报告期，如3月为'一季|1季'；6月为'中'"；9月为'三季|3季'"；12月为'年'"}
     **/
    dateformat: function (str, fmt, def) {
        if (fmt === void 0) { fmt = 'YYYY-MM-DD'; }
        if (def === void 0) { def = '-'; }
        if (this.isEmpty(str)) {
            return def;
        }
        try {
            if (/^\d*$/.test(str)) {
                str = parseInt(str);
            }
            var ret = moment_1["default"](str);
            if (ret.isValid()) {
                var value = ret.format(fmt);
                var o = {
                    "W": weeks[ret.day()],
                    "Q": quarters_z[ret.month()],
                    "q": quarters[ret.month()],
                    "J": quarters0_z[ret.month()],
                    "j": quarters0[ret.month()],
                    "B": reports_z[ret.month()],
                    "b": reports[ret.month()]
                };
                for (var k in o) {
                    if (new RegExp("(" + k + ")").test(value)) {
                        value = value.replace(RegExp.$1, o[k]);
                    }
                }
                return value;
            }
            return def;
        }
        catch (err) {
            return def;
        }
    },
    /**
     * 字符串展示格式化方法
     * @param {string} str:字段值
     * @param {any} fmt：格式化参数，数组或对象
     **/
    strformat: function (str, fmt, def) {
        if (def === void 0) { def = '-'; }
        if (this.isEmpty(str)) {
            return def;
        }
        str = str.toString();
        if (typeof (fmt) == "object") {
            if (fmt.len && fmt.ellipsis) {
                str = this.cutstr(str, fmt.len, fmt.ellipsis, def);
            }
            for (var key in fmt) {
                if (fmt[key] !== undefined) {
                    var reg = new RegExp("({" + key + "})", "g");
                    str = str.replace(reg, fmt[key]);
                }
            }
        }
        else if (Array.isArray(fmt)) {
            fmt.forEach(function (s, i) {
                var reg = new RegExp("({)" + i + "(})", "g");
                str = str.replace(reg, s);
            });
        }
        return str;
    },
    /**
    * js截取字符串，中英文都能用
    * @param {string} str: 需要截取的字符串
    * @param {number} len: 需要截取的长度【中文算两个】
    * @param {string} ellipsis: 溢出文字
    * @returns {string} 截取后的字符串
    */
    cutstr: function (str, len, ellipsis, def) {
        if (def === void 0) { def = '-'; }
        if (this.isEmpty(str)) {
            return def;
        }
        if (typeof ellipsis != "string")
            ellipsis = "...";
        var str_length = 0;
        var str_len = 0;
        var a;
        for (var i = 0; i < str.length; i++) {
            a = str.charAt(i);
            str_length++;
            if (escape(a).length > 4) {
                //中文字符的长度经编码之后大于4  
                str_length++;
            }
            if (str_length <= len) {
                str_len++;
            }
        }
        //如果给定字符串小于指定长度，则返回源字符串；  
        if (str_length <= len) {
            return str.toString();
        }
        else {
            return str.substr(0, str_len).concat(ellipsis);
        }
    },
    /**
    * html标签转义
    * @param {string} str: 需要处理的字符串
    * @returns {string} 处理后的字符串
    */
    escapeHTML: function (text) {
        if (typeof text === 'string') {
            return text
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#039;')
                .replace(/`/g, '&#x60;');
        }
        return text;
    },
    /**
     * 验证数据是否存在
     * @param data
     */
    isEmpty: function (data) {
        if (data === '' || data === '-' || data === undefined || data === null || typeof data === "undefined" || data === false) {
            return true;
        }
        return false;
    },
    /**
     *
     * 分割数据转json,三期|f10
     */
    splitdatatojson: function (data, fields) {
        //console.info(fields);
        var ret = {};
        var datas = [];
        var SplitSymbol = data.SplitSymbol;
        var fs = data.FieldName.split(',');
        if (!fields)
            fields = fs;
        var _loop_1 = function () {
            itemstr = data.Data[i];
            var model = {};
            var arr = itemstr.split(SplitSymbol);
            arr.forEach(function (v, i) {
                var field = fs[i];
                if (fields.indexOf(field) > -1)
                    model[fs[i]] = v;
            });
            datas.push(model);
        };
        var itemstr;
        for (var i = 0; i < data.Data.length; i++) {
            _loop_1();
        }
        Object.keys(data).forEach(function (key) {
            if (key != "Data")
                ret[key] = data[key];
        });
        ret.datas = datas;
        //console.info(ret);
        return ret;
    },
    /**
     *
     * 数据转行情code(仅支持沪深京港)
     * 数据平台接口尽量传对象使用
     * 多层逻辑
     * obj 情况 有 SECURITY_CODE(或其他),TRADE_MARKET_CODE
     * obj 情况 有 SECUCODE
     * obj 情况 只有 SECURITY_CODE
     * string 情况  带市场格式   如   “605199.SH"
     * string 情况  6位数字    如  "300059"
     * string 情况  5位数字    如  "00001"
     *
     *
     * 都不成立的情况用第二个参数（尽量不用）
     */
    datatoquotecode: function (item, scode) {
        if (scode === void 0) { scode = ''; }
        if (typeof item === "object") {
            var code = (item === null || item === void 0 ? void 0 : item.SECURITY_CODE) || (item === null || item === void 0 ? void 0 : item.SECURITYCODE) || (item === null || item === void 0 ? void 0 : item.SCODE) || (item === null || item === void 0 ? void 0 : item.SCode) || (item === null || item === void 0 ? void 0 : item.scode) || (item === null || item === void 0 ? void 0 : item.code) || (item === null || item === void 0 ? void 0 : item.securitycode) || (item === null || item === void 0 ? void 0 : item.securityCode);
            if ((item === null || item === void 0 ? void 0 : item.TRADE_MARKET_CODE) && code.length && code.length == 6) {
                //沪
                if (item.TRADE_MARKET_CODE == "069001001001" || item.TRADE_MARKET_CODE == "069001001003" || item.TRADE_MARKET_CODE == "069001001006") {
                    return "1." + code;
                }
                //深
                else if (item.TRADE_MARKET_CODE == "069001002001" || item.TRADE_MARKET_CODE == "069001002002" || item.TRADE_MARKET_CODE == "069001002005") {
                    return "0." + code;
                }
                //京
                else if (item.TRADE_MARKET_CODE == "069001017") {
                    return "0." + code;
                }
                //新三板，老三版
                else if (item.TRADE_MARKET_CODE == "069001004001" || item.TRADE_MARKET_CODE == "069001004002") {
                    return "0." + code;
                }
            }
            if (item === null || item === void 0 ? void 0 : item.SECUCODE) {
                item = item.SECUCODE;
            }
            else if (code) {
                item = code;
            }
            else {
                item = scode;
            }
        }
        if (typeof item === "string") {
            var codes = item.split('.');
            if (codes.length == 2) {
                var mkt = '';
                if (codes[1].toUpperCase() == "SH")
                    mkt = '1';
                else if (codes[1].toUpperCase() == "SZ")
                    mkt = '0';
                else if (codes[1].toUpperCase() == "BJ")
                    mkt = '0';
                else if (codes[1].toUpperCase() == "NQ")
                    mkt = '0';
                else if (codes[1].toUpperCase() == "HK")
                    mkt = '116';
                return mkt + "." + codes[0];
            }
            if (/\d{6}/.test(item)) {
                item = /\d{6}/.exec(item)[0];
                var one = item.substr(0, 1);
                var three = item.substr(0, 3);
                if (one == "4" || one == "8" || three == "920") {
                    //沪
                    return "0." + item;
                }
                else if (one == "5" || one == "6" || one == "9") {
                    //京
                    return "1." + item;
                }
                else {
                    if (three == "009" || three == "126" || three == "110" || three == "201" || three == "202" || three == "203" || three == "204") {
                        //沪
                        return "1." + item;
                    }
                    else {
                        //深   其它
                        return "0." + item;
                    }
                }
            }
            else if (/^\d{5}/.test(item)) {
                item = /^\d{5}/.exec(item)[0];
                //港
                return "116." + item;
            }
        }
    },
    fmtsname: function (name, len) {
        if (len === void 0) { len = 8; }
        return "<span title=\"" + this.escapeHTML(this.strformat(name)) + "\">" + this.cutstr(name, len) + "</span>";
    },
    getUrlParam: function (name) {
        var reg = new RegExp("(^|&)" + name + "=([^&]*)(&|$)", "i");
        var r = window.location.search.substr(1).match(reg);
        if (r != null)
            return decodeURIComponent(r[2]);
        return null;
    },
    toFixed: function (value, num) {
        var power = Math.pow(10, num);
        return (Math.round(value * power) / power).toFixed(num);
    },
    enhancedXSSCheck: function (input) {
        if (typeof input !== 'string') {
            return null;
        }
        // 检测HTML标签和script标签的正则表达式
        var htmlTagRegex = /<[^>]+>/i;
        var scriptTagRegex = /<script\b[^>]*>([\s\S]*?)<\/script>/gi;
        // 检测到任何HTML标签或script标签就返回null
        if (htmlTagRegex.test(input) || scriptTagRegex.test(input)) {
            console.log(input);
            return false;
        }
        // 安全的内容直接返回
        return true;
        // return xssPatterns.some(pattern => pattern.test(input)) ? false : true;
    }
};


/***/ }),

/***/ 2:
/***/ (function(module, exports) {

function getUrlParam(name){
    var urlpara = location.search;
    var par = {};
    if (urlpara != "") {
      urlpara = urlpara.substring(1, urlpara.length);
      var para = urlpara.split("&");
      var parname;
      var parvalue;
      for (var i = 0; i < para.length; i++) {
        parname = para[i].substring(0, para[i].indexOf("="));
        parvalue = para[i].substring(para[i].indexOf("=") + 1, para[i].length);
        par[parname] = parvalue;
      }
    }
    if(typeof (par[name]) != "undefined"){
      return par[name];
    }
    else{
      return null;
    }
  }

module.exports = {
  development: {
    dataurl: function(){
      // return '//reportapi.uat.emapd.com/'
      return '//reportapi-uat.eastmoney.com/'
    },
    quoteurl: function(){
      return '//push2.eastmoney.com/'
      // return '//push2test.eastmoney.com/'
      return '//push2-uatfj.eastmoney.com/'
    },
    quotehisurl: function(){
      // return '//push2test.eastmoney.com/'
      return '//push2his-uatfj.eastmoney.com/'
    },    
    dcfmurl: function(){
      return '//dcfm.eastmoney.com/' 
    }, 
    anoticeurl: function(){
      return '//np-anotice-stock.eastmoney.com/'
    },
    cnoticeurl: function(){
      return '//np-cnotice-stock-test.eastmoney.com/'
    },
    datacenter: function(){
      // return '//datacenter-web.eastmoney.com/'
      return '//testdatacenter.eastmoney.com/'
      // return '//betadatacenter.eastmoney.com/'
      let env = getUrlParam('myenv') || '';
      if(env == 'prod') {
        return '//datacenter-web.eastmoney.com/'
      }
      return '//betadatacenter.eastmoney.com/'
      // return '//testdatacenter.eastmoney.com/'
    },
    datacenter2: function(){
      // return '//datacenter-web.eastmoney.com/'
      return '//testdatacenter.eastmoney.com/'
      // return '//betadatacenter.eastmoney.com/'
      // let env = getUrlParam('myenv') || '';
      // if(env == 'prod') {
      //   return '//datacenter-web.eastmoney.com/'
      // }
      // return '//testdatacenter.eastmoney.com/'
    },
    soapi: function(){
      return '//searchapi.eastmoney.com/'
    }, 
    cmsdataapi: function(){
      return '//cmsdataapi.eastmoney.com/'
    },
    newsinfo: function(){
      return '//newsinfo.eastmoney.com/'
    },
    url: function(){
      return ''
    },
    cmsapi : function () {
      return '//np-listapi.eastmoney.com//'
    }
  },
  zptest: {
    dataurl: function(){
      // return '//reportapi.eastmoney.com/'
      // return '//reportapi.uat.emapd.com/'
      return '//reportapi-uat.eastmoney.com/'

    },
    quoteurl: function(){
      return '//push2.eastmoney.com/'
      // return '//push2test.eastmoney.com/'
      return '//push2-uatfj.eastmoney.com/'
    },
    quotehisurl: function(){
      // return '//push2test.eastmoney.com/'
      return '//push2his-uatfj.eastmoney.com/'
    }, 
    dcfmurl: function(){
      return '//dcfm.eastmoney.com/'
    },
    anoticeurl: function(){
      return '//np-anotice-stock-test.eastmoney.com/'
    },
    cnoticeurl: function(){
      // return '//np-cnotice-stock-test.emapd.com/'
      return '//np-cnotice-stock-test.eastmoney.com/'
    },
    datacenter: function(){
      return '//testdatacenter.eastmoney.com/'
      // return '//betadatacenter.eastmoney.com/'
      // return '//datacenter-web.eastmoney.com/'
      let env = getUrlParam('myenv') || '';
      if(env == 'prod') {
        return '//datacenter-web.eastmoney.com/'
      }
      return '//betadatacenter.eastmoney.com/'
      // return '//testdatacenter.eastmoney.com/'
    },
    datacenter2: function(){
      // return '//datacenter-web.eastmoney.com/'
      return '//testdatacenter.eastmoney.com/'
      // return '//betadatacenter.eastmoney.com/'
      // let env = getUrlParam('myenv') || '';
      // if(env == 'prod') {
      //   return '//datacenter-web.eastmoney.com/'
      // }
      // return '//testdatacenter.eastmoney.com/'
    },
    soapi: function(){
      return '//searchapi.eastmoney.com/'
    },    
    cmsdataapi: function(){
      return '//cmsdataapi.eastmoney.com/'
    },
    newsinfo: function(){
      return '//newsinfo.eastmoney.com/'
    },
    url: function(){
      return ''
    },
    cmsapi : function () {
      return '//np-listapi.eastmoney.com//'
    }
  },
  gray: {
    dataurl: function(){
      return '//reportapi.eastmoney.com/'
    },
    quoteurl: function(){
      return '//push2.eastmoney.com/'
    },
    quotehisurl: function(){
      return '//push2his-uatfj.eastmoney.com/'
    },      
    dcfmurl: function(){
      return '//dcfm.eastmoney.com/'
    },
    anoticeurl: function(){
      return '//np-anotice-stock.eastmoney.com/'
    },
    cnoticeurl: function(){
      return '//np-cnotice-stock.eastmoney.com/'
    },
    datacenter: function(){
      return '//graydatacenter.eastmoney.com/web/'
    },
    datacenter2: function(){
      return '//graydatacenter.eastmoney.com/web/'
    },
    soapi: function(){
      return '//searchapi.eastmoney.com/'
    },      
    cmsdataapi: function(){
      return '//cmsdataapi.eastmoney.com/'
    },
    newsinfo: function(){
      return '//newsinfo.eastmoney.com/'
    },
    url: function(){
      return ''
    },
    cmsapi : function () {
      return '//np-listapi.eastmoney.com//'
    }
  },
  production: {
    dataurl: function(){
      return '//reportapi.eastmoney.com/'
    },
    quoteurl: function(){
      return '//push2.eastmoney.com/'
    },
    quotehisurl: function(){
      return '//push2his.eastmoney.com/'
    },      
    dcfmurl: function(){
      return '//dcfm.eastmoney.com/'
    },
    anoticeurl: function(){
      return '//np-anotice-stock.eastmoney.com/'
    },
    cnoticeurl: function(){
      return '//np-cnotice-stock.eastmoney.com/'
    },
    datacenter: function(){
      return '//datacenter-web.eastmoney.com/'
    },
    datacenter2: function(){
      return '//datacenter-web.eastmoney.com/'
    },
    soapi: function(){
      return '//searchapi.eastmoney.com/'
    },      
    cmsdataapi: function(){
      return '//cmsdataapi.eastmoney.com/'
    },
    newsinfo: function(){
      return '//newsinfo.eastmoney.com/'
    },
    url: function(){
      return ''
    },
    cmsapi : function () {
      return '//np-listapi.eastmoney.com//'
    }
  },
  getParam: function(name){
    var urlpara = location.search;
    var par = {};
    if (urlpara != "") {
      urlpara = urlpara.substring(1, urlpara.length);
      var para = urlpara.split("&");
      var parname;
      var parvalue;
      for (var i = 0; i < para.length; i++) {
        parname = para[i].substring(0, para[i].indexOf("="));
        parvalue = para[i].substring(para[i].indexOf("=") + 1, para[i].length);
        par[parname] = parvalue;
      }
    }
    if(typeof (par[name]) != "undefined"){
      return par[name];
    }
    else{
      return null;
    }
  },
  getWebPath: function (name) {
    var env=(window.service&&service.ENV)||this.getParam('env');

    // env = "production"

    if (env&&this[env]) {
      return this[env][name]()
    }
    return this.production[name]()
  }
}



/***/ }),

/***/ 23:
/***/ (function(module, exports, __webpack_require__) {

"use strict";

var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
exports.__esModule = true;
/**
 * 行业研报和个股研报参数配置
 *
 * */
var utils_1 = __importDefault(__webpack_require__(3));
/**
 * 行业研报和个股研报参数说明：
 * qType:必填，，0个股研报，1行业研报，2策略报告，3宏观研究，4券商晨会
 * reportType研报类型，不查询填*或不带该字段  个股研报是2 行业研报是3(可不要)
 *
 *  */
var dataurl = {
    "industry": {
        url: 'report/list?cb=?',
        params: {
            industryCode: '*',
            pageSize: 50,
            industry: '*',
            rating: '*',
            ratingChange: '*',
            beginTime: '',
            endTime: '',
            pageNo: 1,
            fields: "",
            qType: 1
        },
        node_params: function () {
            return {
                industryCode: '*',
                pageSize: 50,
                industry: '*',
                rating: '',
                ratingchange: '*',
                beginTime: utils_1["default"].getDateStr(-2, 0, 0),
                endTime: utils_1["default"].getDateStr(0, 0, 0),
                pageNo: 1,
                fields: '',
                qType: 1,
                orgCode: '',
                code: '*',
                rcode: ''
            };
        }
    },
    // code=*&reportType=2&beginTime=20101101&endTime=2019-12-02%2011:44:44&start=0&rows=1&delay=365&cb=showData&qType=0
    "stock": {
        url: 'report/list?cb=?',
        params: {
            industryCode: '*',
            pageSize: 50,
            industry: '*',
            rating: '*',
            ratingChange: '*',
            beginTime: '',
            endTime: '',
            pageNo: 1,
            fields: "",
            qType: 0 //必填，0标识查询个股研报，1标识行业研报
        },
        node_params: function () {
            return {
                industryCode: '*',
                pageSize: 50,
                industry: '*',
                rating: '',
                beginTime: utils_1["default"].getDateStr(-2, 0, 0),
                endTime: utils_1["default"].getDateStr(0, 0, 0),
                pageNo: 1,
                fields: '',
                qType: 0,
                ratingChange: '',
                orgCode: '',
                code: '*',
                rcode: ''
            };
        }
    },
    "profitforecast": {
        hostname: 'datacenter',
        url: 'api/data/v1/get',
        params: {
            reportName: 'RPT_WEB_RESPREDICT',
            columns: 'WEB_RESPREDICT',
            pageNumber: 1,
            pageSize: 50,
            sortTypes: -1,
            sortColumns: 'RATING_ORG_NUM'
        },
        node_params: function () {
            return {
                reportName: 'RPT_WEB_RESPREDICT',
                columns: 'WEB_RESPREDICT',
                pageNumber: 1,
                pageSize: 50,
                sortTypes: -1,
                sortColumns: 'RATING_ORG_NUM'
            };
        }
    },
    //indexnewreport  //首页:公司研究：最新研报：取的是个股研报前五条
    "indexnewreport": {
        url: 'report/list?cb=?',
        params: {
            industryCode: '*',
            pageSize: 5,
            industry: '*',
            rating: '*',
            ratingchange: '*',
            //reportType: 2, //研报类型，不查询填*或不带该字段  个股研报是2 行业研报是3
            // isNew:'002',
            beginTime: '',
            endTime: '',
            pageNo: 1,
            fields: "",
            qType: 0 //必填，0标识查询个股研报，1标识行业研报
        },
        node_params: function () {
            return {
                industryCode: '*',
                pageSize: 5,
                industry: '*',
                rating: '*',
                ratingchange: '*',
                beginTime: utils_1["default"].getDateStr(-2, 0, 0),
                endTime: utils_1["default"].getDateStr(0, 0, 0),
                pageNo: 1,
                fields: "",
                qType: 0 //必填，0标识查询个股研报，1标识行业研报
            };
        }
    },
    //industryresearch 首页行业研究
    "industryresearch": {
        url: 'report/list?cb=?',
        params: {
            industryCode: '*',
            pageSize: 5,
            industry: '*',
            rating: '*',
            ratingchange: '*',
            columnCode: "002002",
            //reportType: 3, //研报类型，不查询填*或不带该字段  个股研报是2 行业研报是3
            beginTime: utils_1["default"].getDateStr(-2, 0, 0),
            endTime: utils_1["default"].getDateStr(0, 0, 0),
            pageNo: 1,
            fields: "",
            qType: 1
        },
        node_params: function () {
            return {
                industryCode: '*',
                pageSize: 5,
                industry: '*',
                rating: '*',
                ratingchange: '*',
                columnCode: "002002",
                //reportType: 3, //研报类型，不查询填*或不带该字段  个股研报是2 行业研报是3
                beginTime: '',
                endTime: '',
                pageNo: 1,
                fields: "",
                qType: 1
            };
        }
    },
    //indexylyc 首页盈利预测
    "indexylyc": {
        hostname: 'datacenter',
        url: 'api/data/v1/get',
        params: {
            reportName: 'RPT_WEB_RESPREDICT',
            columns: 'WEB_RESPREDICT',
            pageNumber: 1,
            pageSize: 5,
            sortTypes: -1,
            sortColumns: 'RATING_ORG_NUM'
        },
        node_params: function () {
            return {
                reportName: 'RPT_WEB_RESPREDICT',
                columns: 'WEB_RESPREDICT',
                pageNumber: 1,
                pageSize: 5,
                sortTypes: -1,
                sortColumns: 'RATING_ORG_NUM'
            };
        }
    },
    //首页宏观研究indexmacresearch
    "indexmacresearch": {
        url: 'report/jg?cb=?',
        params: {
            pageSize: 5,
            beginTime: '',
            endTime: '',
            pageNo: 1,
            fields: "",
            qType: 3 // 必填，2策略报告，3宏观研究，4券商晨会
        },
        node_params: function () {
            return {
                pageSize: 5,
                beginTime: '',
                endTime: '',
                pageNo: 1,
                fields: "",
                qType: 3 // 必填，2策略报告，3宏观研究，4券商晨会
            };
        }
    },
    //首页新股研究  indexnewstock
    "indexnewstock": {
        url: 'report/newStockList?cb=?',
        params: {
            pageSize: 5,
            beginTime: '',
            endTime: '',
            pageNo: 1,
            fields: "",
            qType: 4 // 必填，2策略报告，3宏观研究，4券商晨会
        },
        node_params: function () {
            return {
                pageSize: 5,
                beginTime: '',
                endTime: '',
                pageNo: 1,
                fields: "",
                qType: 4 // 必填，2策略报告，3宏观研究，4券商晨会
            };
        }
    },
    //首页策略报告
    "indexstrategyreport": {
        url: 'report/jg?cb=?',
        params: {
            pageSize: 5,
            beginTime: '',
            endTime: '',
            pageNo: 1,
            fields: "",
            qType: 2 // 必填，2策略报告，3宏观研究，4券商晨会
        },
        node_params: function () {
            return {
                pageSize: 5,
                beginTime: '',
                endTime: '',
                pageNo: 1,
                fields: "",
                qType: 2 // 必填，2策略报告，3宏观研究，4券商晨会
            };
        }
    },
    //首页券商晨会 indexbroker
    "indexbroker": {
        url: 'report/jg?cb=?',
        params: {
            pageSize: 5,
            beginTime: '',
            endTime: '',
            pageNo: 1,
            fields: "",
            qType: 4 // 必填，2策略报告，3宏观研究，4券商晨会
        },
        node_params: function () {
            return {
                pageSize: 5,
                beginTime: '',
                endTime: '',
                pageNo: 1,
                fields: "",
                qType: 4 // 必填，2策略报告，3宏观研究，4券商晨会
            };
        }
    },
    //首页东财分析师  eastmoneyanalyst
    "eastmoneyanalyst": {
        url: '',
        params: {}
    },
    //新股研报
    "newstock": {
        //以下是更换的新接口i
        url: 'report/newStockList?cb=?',
        params: {
            pageSize: 50,
            beginTime: '',
            endTime: '',
            pageNo: 1,
            fields: "",
            qType: 4 // 必填，2策略报告，3宏观研究，4券商晨会
        },
        node_params: function () {
            return {
                pageSize: 50,
                beginTime: utils_1["default"].getDateStr(-2, 0, 0),
                endTime: utils_1["default"].getDateStr(0, 0, 0),
                pageNo: 1,
                fields: "",
                qType: 4 // 必填，2策略报告，3宏观研究，4券商晨会
            };
        }
    },
    //宏观研报 
    "macresearch": {
        url: 'report/jg?cb=?',
        params: {
            pageSize: 50,
            beginTime: '',
            endTime: '',
            pageNo: 1,
            fields: "",
            qType: 3 // 必填，2策略报告，3宏观研究，4券商晨会
        },
        node_params: function () {
            return {
                orgCode: '',
                author: '',
                pageSize: 50,
                beginTime: utils_1["default"].getDateStr(-2, 0, 0),
                endTime: utils_1["default"].getDateStr(0, 0, 0),
                pageNo: 1,
                fields: "",
                qType: 3 // 必填，2策略报告，3宏观研究，4券商晨会
            };
        }
    },
    //策略报告 
    "strategyreport": {
        url: 'report/jg?cb=?',
        params: {
            pageSize: 50,
            beginTime: '',
            endTime: '',
            pageNo: 1,
            fields: "",
            qType: 2 // 必填，2策略报告，3宏观研究，4券商晨会
        },
        node_params: function () {
            return {
                pageSize: 50,
                beginTime: utils_1["default"].getDateStr(-2, 0, 0),
                endTime: utils_1["default"].getDateStr(0, 0, 0),
                pageNo: 1,
                fields: "",
                qType: 2 // 必填，2策略报告，3宏观研究，4券商晨会
            };
        }
    },
    //券商晨报 
    "brokerreport": {
        url: 'report/jg?cb=?',
        params: {
            pageSize: 50,
            beginTime: '',
            endTime: '',
            pageNo: 1,
            fields: "",
            qType: 4 // 必填，2策略报告，3宏观研究，4券商晨会
        },
        node_params: function () {
            return {
                pageSize: 50,
                beginTime: utils_1["default"].getDateStr(-2, 0, 0),
                endTime: utils_1["default"].getDateStr(0, 0, 0),
                pageNo: 1,
                fields: "",
                qType: 4 // 必填，2策略报告，3宏观研究，4券商晨会
            };
        }
    },
    //机构发布研报：国盛证券发布研报
    "orgpublish": {
        url: 'report/dg?cb=?',
        params: {
            pageNo: 1,
            pageSize: 50,
            author: '*',
            orgCode: '',
            beginTime: '',
            endTime: '',
            // qType:0
            qType: ''
            // beginTime: util.GetDayStr(2),
            // endTime: util.GetDayStr(0)
        }
    },
    //作者个人 
    "personalpublish": {
        url: 'report/dg?cb=?',
        params: {
            pageNo: 1,
            pageSize: 50,
            author: '',
            orgCode: '',
            beginTime: '',
            endTime: '',
            // qType:0
            qType: ""
        }
    },
    //单个个股 
    "singlestock": {
        url: 'report/list?cb=?',
        params: {
            pageNo: 1,
            pageSize: 50,
            code: '*',
            industryCode: '*',
            industry: '*',
            rating: '*',
            ratingchange: '*',
            beginTime: '',
            endTime: '',
            fields: "",
            qType: 0 //必填，0标识查询个股研报，1标识行业研报
        }
    },
    //策略报告正文：策略报告
    "zw_stratyge": {
        url: 'report/jg?cb=?',
        params: {
            pageSize: 5,
            beginTime: '',
            endTime: '',
            pageNo: 1,
            fields: "",
            qType: 2 // 必填，2策略报告，3宏观研究，4券商晨会
        }
    },
    //策略报告正文：最新研究报告
    "zw_neweast": {
        url: 'report/dg?cb=?',
        params: {
            pageNo: 1,
            pageSize: 5,
            author: '*',
            orgCode: '*',
            qType: '*',
            beginTime: '',
            endTime: ''
        }
    },
    //策略报告正文：热门行业追踪:行业研报接口
    "zw_hotindustry": {
        url: 'report/list?cb=?',
        params: {
            industryCode: '*',
            pageSize: 15,
            industry: '*',
            rating: '*',
            ratingchange: '*',
            beginTime: '',
            endTime: '',
            pageNo: 1,
            fields: "",
            qType: 1 //必填，0个股研报，1行业研报，2策略报告，3宏观研究，4券商晨会
        }
    },
    // 行业研报正文：文中涉及到的个股
    "zw_industry_stock": {
        url: 'report/list?cb=?',
        params: {
            code: '',
            industryCode: '*',
            pageSize: 5,
            industry: '*',
            rating: '*',
            ratingchange: '*',
            beginTime: '',
            endTime: '',
            pageNo: 1,
            fields: "",
            qType: 0 //必填，0标识查询个股研报，1标识行业研报
        }
    },
    //行业研报正文 ：所在行业研究报告
    "zw_indsneweast": {
        url: 'report/list?cb=?',
        params: {
            industryCode: '*',
            pageSize: 5,
            industry: '*',
            rating: '*',
            ratingchange: '*',
            beginTime: '',
            endTime: '',
            pageNo: 1,
            fields: "",
            qType: 1
        }
    },
    //行业研报正文 ：所在行业热门个股评级一览
    "zw_indshotstock": {},
    //行业研报正文 ：所在行业个股未来3年盈利预测
    "zw_indsstockforecast": {
        hostname: 'datacenter',
        url: 'api/data/v1/get',
        params: {
            reportName: 'RPT_WEB_RESPREDICT',
            columns: 'WEB_RESPREDICT',
            pageNumber: 1,
            pageSize: 15,
            sortTypes: -1,
            sortColumns: 'RATING_ORG_NUM'
        }
    },
    //行业研报正文  ：行业个股财务指标排行榜
    "zw_indggcw_table": {
        url: '',
        params: {}
    },
    //行业研报正文 ：热门行业追踪
    "zw_indshotindustry": {
    // url: 'report/list?cb=?',
    // params: {
    //   industryCode: '*',
    //   pageSize: 15,
    //   industry: '*',//行业
    //   rating: '*',
    //   ratingchange: '*',//评级变化，不查询填*或不带该字段
    //   beginTime: '',
    //   endTime: '',
    //   pageNo: 1,
    //   fields: "",
    //   qType: 1       //必填，0个股研报，1行业研报，2策略报告，3宏观研究，4券商晨会
    // }
    },
    //宏观研究正文
    "zw_macsearch": {
        url: 'report/jg?cb=?',
        params: {
            pageSize: 5,
            beginTime: '',
            endTime: '',
            pageNo: 1,
            fields: "",
            qType: 3 // 必填，2策略报告，3宏观研究，4券商晨会
        }
    },
    //券商晨会正文
    "zw_broker": {
        url: 'report/jg?cb=?',
        params: {
            pageSize: 5,
            beginTime: '',
            endTime: '',
            pageNo: 1,
            fields: "",
            qType: 4 // 必填，2策略报告，3宏观研究，4券商晨会
        }
    },
    "zw_neweastbroker": {
        url: 'report/dg?cb=?',
        params: {
            pageNo: 1,
            pageSize: 5,
            author: '*',
            orgCode: '*',
            qType: '*',
            beginTime: '',
            endTime: ''
        }
    },
    //单个个股正文 :行业最新
    "zw_indsneweast_stock": {
        url: 'report/list?cb=?',
        params: {
            industryCode: '*',
            pageSize: 5,
            industry: '*',
            rating: '*',
            ratingchange: '*',
            beginTime: '',
            endTime: '',
            pageNo: 1,
            fields: "",
            qType: 1
        }
    },
    "zw_orgneweast_stock": {
        url: 'report/list?cb=?',
        params: {
            industryCode: '*',
            pageSize: 5,
            orgCode: '',
            industry: '*',
            rating: '*',
            ratingChange: '*',
            beginTime: utils_1["default"].getDateStr(-2, 0, 0),
            endTime: utils_1["default"].getDateStr(0, 0, 0),
            pageNo: 1,
            fields: "",
            qType: 0 //0个股1行业
        }
        // url: 'report/dg?cb=?',
        // params: {
        //   pageNo: 1,//页码数
        //   pageSize: 5,//每页条数
        //   author: '*',//作者 例：11000175403.潘红敏
        //   orgCode: '', //机构代码 
        //   beginTime: '',
        //   endTime: '',
        //   qType: ''
        // }
    },
    //单个个股的最新  zw_stockneweast_stock
    "zw_stockneweast_stock": {
        url: 'report/list?cb=?',
        params: {
            code: '',
            industryCode: '*',
            pageSize: 5,
            industry: '*',
            rating: '*',
            ratingchange: '*',
            beginTime: '',
            endTime: '',
            pageNo: 1,
            fields: "",
            qType: 0 //0个股1行业
        }
    },
    //个股研报正文 ：所在行业热门个股评级一览
    "zw_indshotstockpj": {}
};
exports["default"] = dataurl;


/***/ }),

/***/ 24:
/***/ (function(module, exports, __webpack_require__) {

"use strict";

var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
exports.__esModule = true;
/**
 * 表格配置
 * paramname 排序必须字段
 *
 */
var utils_1 = __importDefault(__webpack_require__(3));
var fieldtools_1 = __importDefault(__webpack_require__(1));
//评级变动配置
function getRatingChangeStr(ratingChange) {
    var ratingChangeName = "";
    switch (ratingChange + '') {
        case '0':
            ratingChangeName = '调高';
            break;
        case '1':
            ratingChangeName = '调低';
            break;
        case '2':
            ratingChangeName = '首次';
            break;
        case '3':
            ratingChangeName = '维持';
            break;
        case '4':
            ratingChangeName = '无';
            break;
        default:
            ratingChangeName = '-';
            break;
    }
    return ratingChangeName;
}
;
//作者名字处理:len需要截取的字符串长度2个单位等于一个汉字
function authorFormat(v, len) {
    var authorInfo = v.author ? v.author : '';
    var anthorStr = '';
    if (authorInfo != '' && authorInfo.length > 0) {
        for (var i = 0; i < authorInfo.length; i++) {
            var info = authorInfo[i] || [];
            // let name = authorInfo[i] ? authorInfo[i].split('.')[1] : '';
            var name_1 = (info === null || info === void 0 ? void 0 : info.replace(/^\d+./, '')) || "";
            if (len) {
                name_1 = utils_1["default"].txtLeft(name_1, len, true);
            }
            var id = info ? info.split('.')[0] : '';
            anthorStr += "<a class=\"tt-wrap\" href=\"personalpublish.jshtml?authorid=" + id + "&orgcode=" + v.orgCode + ("\">" + name_1 + "</a>&nbsp;");
        }
    }
    return anthorStr;
}
;
//单个机构和单个作者正文跳转地址
function getZWUrl(v) {
    if (v.columnType == '个股研报') {
        var date = ((v.publishDate).split(' ')[0]).replace(/-/g, '');
        return "<a href=\"/report/info/" + v.infoCode + (".html\">" + v.title + "</a>");
    }
    ;
    if (v.columnType == '行业研报') {
        return "<a href=\"/report/zw_industry.jshtml?infocode=" + v.infoCode + ("\">" + v.title + "</a>");
    }
    ;
    if (v.columnType == '券商晨会') {
        return "<a href=\"/report/zw_brokerreport.jshtml?encodeUrl=" + v.encodeUrl + ("\">" + v.title + "</a>");
    }
    ;
    if (v.columnType == '宏观研究' || v.columnType == '宏观策略') {
        return "<a href=\"/report/zw_macresearch.jshtml?encodeUrl=" + v.encodeUrl + ("\">" + v.title + "</a>");
    }
    ;
    if (v.columnType == '策略报告') {
        return "<a href=\"/report/zw_strategy.jshtml?encodeUrl=" + v.encodeUrl + ("\">" + v.title + "</a>");
    }
}
;
//单个机构和单个作者类型跳转地址
function geTypeUrl(v) {
    if (v.columnType == '个股研报') {
        return "<a href=\"/report/stock.jshtml\">" + v.columnType + "</a>";
    }
    ;
    if (v.columnType == '行业研报') {
        return "<a href=\"/report/industry.jshtml\">" + v.columnType + "</a>";
    }
    ;
    if (v.columnType == '券商晨会') {
        return "<a href=\"/report/brokerreport.jshtml\">" + v.columnType + "</a>";
    }
    ;
    if (v.columnType == '宏观研究' || v.columnType == '宏观策略') {
        return "<a href=\"/report/macresearch.jshtml\">" + v.columnType + "</a>";
    }
    ;
    if (v.columnType == '策略报告') {
        return "<a href=\"/report/strategyreport.jshtml\">" + v.columnType + "</a>";
    }
}
;
//单个机构和单个作者研究对象取值和跳转地址
function getYJDXUrl(v) {
    if (v.columnType == '个股研报') {
        if (v.stockCode != '' && v.stockName != '') {
            return "<a href=\"/report/" + v.stockCode + ".html\">" + v.stockName + "(" + v.stockCode + ")</a>";
        }
    }
    else if (v.columnType == '行业研报') { //产品确认：跳转只带行业代码即可，不带机构
        if (v.industryName != '' && v.industryCode != '') {
            return "<a href=\"/report/industry.jshtml?hyid=" + v.industryCode + ("\">" + v.industryName + "</a>");
        }
    }
    else {
        return '';
    }
}
;
var config = {
    //首页行业研究
    "industryresearch": {
        config: [
            {
                width: 44,
                th: function name(v) {
                    return '序号';
                },
                td: function (v) {
                    return v.index + 1;
                }
            },
            {
                th: function name(v) {
                    return '行业名称';
                },
                td: function (v) {
                    var bkcode = v.industryCode ? v.industryCode > 999 ? 'bk' + v.industryCode : 'bk0' + v.industryCode : '';
                    return "<a href=\"//quote.eastmoney.com/center/boardlist.html#boards-" + bkcode + ("\">" + v.industryName + "</a>");
                }
            },
            {
                th: function name(v) {
                    return '涨跌幅';
                },
                td: function (v) {
                    return '<span class="hqspan" data-hqtype="zdf" data-code="' + v.industryCode + '"></span>';
                }
            },
            {
                width: 130,
                th: function name(v) {
                    return '相关';
                },
                td: function (v) {
                    var bkcode = v.industryCode ? v.industryCode > 999 ? 'bk' + v.industryCode : 'bk0' + v.industryCode : '';
                    return "<a href=\"/report/industry.jshtml?hyid=" + v.industryCode + "\" class=\"red\">\u8BE6\u7EC6</a>&nbsp;\n                  <a href=\"/bkzj/" + v.industryCode + ".html\">\u8D44\u91D1\u6D41</a>&nbsp;\n                  <a href=\"//guba.eastmoney.com/list," + bkcode + ".html\">\u80A1\u5427</a>&nbsp;\n                  <a href=\"//stock.eastmoney.com/hangye/hy" + v.industryCode + ".html\">\u4E13\u533A</a>";
                }
            },
            {
                width: 220,
                th: function name(v) {
                    return '报告名称';
                },
                td: function (v) {
                    return "<a href=\"/report/zw_industry.jshtml?infocode=" + v.infoCode + ("\">" + v.title + "</a>");
                }
            },
            {
                th: function name(v) {
                    return '东财<br>评级';
                },
                td: function (v) {
                    var emRatingName = v.emRatingName ? v.emRatingName : '-';
                    return "" + emRatingName;
                },
                icons: {
                    name: 'faq',
                    title: "\u6839\u636E\u673A\u6784\u62AB\u9732\u7684\u8BC4\u7EA7\u8BF4\u660E\u5212\u5206\u4E1C\u8D22\u8BC4\u7EA7\u6807\u51C6"
                }
            },
            {
                th: function name(v) {
                    return '评级<br>变动';
                },
                td: function (v) {
                    var ratingChangeName = getRatingChangeStr(v.ratingChange);
                    return "" + ratingChangeName;
                }
            },
            {
                th: function name(v) {
                    return '机构名称';
                },
                td: function (v) {
                    return "<a href=\"/report/orgpublish.jshtml?orgcode=" + v.orgCode + ("\">" + (v.orgSName ? v.orgSName : v.orgName) + "</a>");
                }
            },
            {
                paramname: 'count',
                th: function name(v) {
                    return '近一月行<br>业研报数';
                },
                td: function (v) {
                    return "" + v.count;
                }
            },
            {
                th: function name(v) {
                    return '日期';
                },
                td: function (v) {
                    var date = (v.publishDate).split(' ')[0];
                    return date;
                }
            }
        ]
    },
    "indexnewreport": {
        theadfix: {
            thisyear_ylyc: function (v) {
                var currentYear = v.currentYear ? v.currentYear : '';
                return currentYear + '盈利预测';
            },
            nextyear_ylyc: function (v) {
                var nextyear = v.currentYear ? v.currentYear + 1 : '';
                return nextyear + '盈利预测';
            }
        },
        config: [
            {
                width: 44,
                th: function name(v) {
                    return '序号';
                },
                td: function (v) {
                    return v.index + 1;
                }
            },
            {
                width: 60,
                paramname: 'stockCode',
                th: function name(v) {
                    return "\u80A1\u7968\u4EE3\u7801";
                },
                defaultsortType: '',
                td: function (v) {
                    return "<a href=\"//quote.eastmoney.com/unify/r/" + fieldtools_1["default"].datatoquotecode(v.stockCode) + "\">" + v.stockCode + "</a>";
                }
            },
            {
                width: 60,
                th: function name(v) {
                    return '股票简称';
                },
                td: function (v) {
                    return "<a href=\"/stockdata/" + v.stockCode + (".html\">" + fieldtools_1["default"].fmtsname(v.stockName) + "</a>");
                }
            },
            {
                width: 60,
                th: function name(v) {
                    return '相关';
                },
                td: function (v) {
                    return "<a href=\"" + v.stockCode + ".html\" class=\"red\">\u8BE6\u7EC6</a>&nbsp;\n                  <a href=\"//guba.eastmoney.com/list," + v.stockCode + ".html\">\u80A1\u5427</a>";
                }
            },
            {
                width: 130,
                th: function name(v) {
                    return '报告名称';
                },
                td: function (v) {
                    var date = ((v.publishDate).split(' ')[0]).replace(/-/g, '');
                    return "<a href=\"/report/info/" + v.infoCode + (".html\">" + v.title + "</a>");
                }
            },
            {
                th: function name(v) {
                    return '东财<br>评级';
                },
                td: function (v) {
                    var emRatingName = v.emRatingName ? v.emRatingName : '-';
                    return "" + emRatingName;
                },
                icons: {
                    name: 'faq',
                    title: "\u6839\u636E\u673A\u6784\u62AB\u9732\u7684\u8BC4\u7EA7\u8BF4\u660E\u5212\u5206\u4E1C\u8D22\u8BC4\u7EA7\u6807\u51C6"
                }
            },
            {
                th: function name(v) {
                    return '评级<br>变动';
                },
                td: function (v) {
                    var ratingChangeName = getRatingChangeStr(v.ratingChange);
                    return "" + ratingChangeName;
                }
            },
            {
                width: 60,
                th: function name(v) {
                    return '机构';
                },
                td: function (v) {
                    return "<a href=\"/report/orgpublish.jshtml?orgcode=" + v.orgCode + ("\">" + (v.orgSName ? v.orgSName : v.orgName) + "</a>");
                }
            },
            {
                paramname: 'count',
                th: function name(v) {
                    return '近一月<br>个股研<br>报数';
                },
                td: function (v) {
                    return "" + v.count;
                }
            },
            {
                th: function name(v) {
                    return '收益';
                },
                theadfix: 'thisyear_ylyc',
                td: function (v) {
                    var predictThisYearEps = "";
                    if (v.predictThisYearEps != undefined) {
                        predictThisYearEps = v.predictThisYearEps == '' ? '' : Number(v.predictThisYearEps).toFixed(3);
                    }
                    return predictThisYearEps;
                }
            },
            {
                th: function name(v) {
                    return '市盈率';
                },
                theadfix: 'thisyear_ylyc',
                td: function (v) {
                    var predictThisYearPe = "";
                    if (v.predictThisYearPe != undefined) {
                        predictThisYearPe = v.predictThisYearPe == '' ? '' : Number(v.predictThisYearPe).toFixed(2);
                    }
                    return predictThisYearPe;
                }
            },
            {
                th: function name(v) {
                    return '收益';
                },
                theadfix: 'nextyear_ylyc',
                td: function (v) {
                    var predictNextYearEps = "";
                    if (v.predictNextYearEps != undefined) {
                        predictNextYearEps = v.predictNextYearEps == '' ? '' : Number(v.predictNextYearEps).toFixed(3);
                    }
                    return predictNextYearEps;
                }
            },
            {
                th: function name(v) {
                    return '市盈率';
                },
                theadfix: 'nextyear_ylyc',
                td: function (v) {
                    var predictNextYearPe = "";
                    if (v.predictNextYearPe != undefined) {
                        predictNextYearPe = v.predictNextYearPe == '' ? '' : Number(v.predictNextYearPe).toFixed(2);
                    }
                    return predictNextYearPe;
                }
            },
            {
                width: 60,
                th: function name(v) {
                    return '行业';
                },
                td: function (v) {
                    var indvInduName = v.indvInduName ? v.indvInduName : '';
                    return "<a href=\"/report/industry.jshtml?hyid=" + v.indvInduCode + ("\">" + indvInduName + "</a>");
                }
            },
            {
                width: 65,
                paramname: 'publishDate',
                th: function name(v) {
                    return '日期';
                },
                td: function (v) {
                    var date = (v.publishDate).split(' ')[0];
                    return date;
                }
            }
        ]
    },
    "indexylyc": {
        theadfix: {
            jgtzpj: function (v) {
                return '机构投资评级(近六个月)';
            }
        },
        config: [
            {
                width: 44,
                th: function name(v) {
                    return '序号';
                },
                td: function (v) {
                    return v.index + 1;
                }
            },
            {
                paramname: 'stockCode',
                th: function name(v) {
                    return '代码';
                },
                td: function (v) {
                    return "<a href=\"//quote.eastmoney.com/unify/r/" + fieldtools_1["default"].datatoquotecode(v) + "\">" + v.SECURITY_CODE + "</a>";
                }
            },
            {
                th: function name(v) {
                    return '名称';
                },
                td: function (v) {
                    return "<a href=\"/stockdata/" + v.SECURITY_CODE + (".html\">" + fieldtools_1["default"].fmtsname(v.SECURITY_NAME_ABBR) + "</a>");
                }
            },
            {
                th: function name(v) {
                    return '相关';
                },
                td: function (v) {
                    return "<a href=\"/report/" + v.SECURITY_CODE + ".html\" class=\"red\">\u8BE6\u7EC6</a>&nbsp;\n                  <a href=\"//guba.eastmoney.com/list," + v.SECURITY_CODE + ".html\">\u80A1\u5427</a>";
                }
            },
            {
                paramname: 'total',
                th: function name(v) {
                    return '研报数';
                },
                td: function (v) {
                    return fieldtools_1["default"].isEmpty(v.RATING_ORG_NUM) ? 0 : v.RATING_ORG_NUM;
                }
            },
            {
                th: function name(v) {
                    return '买入';
                },
                theadfix: 'jgtzpj',
                td: function (v) {
                    return fieldtools_1["default"].isEmpty(v.RATING_BUY_NUM) ? 0 : v.RATING_BUY_NUM;
                }
            },
            {
                th: function name(v) {
                    return '增持';
                },
                theadfix: 'jgtzpj',
                td: function (v) {
                    return fieldtools_1["default"].isEmpty(v.RATING_ADD_NUM) ? 0 : v.RATING_ADD_NUM;
                }
            },
            {
                th: function name(v) {
                    return '中性';
                },
                theadfix: 'jgtzpj',
                td: function (v) {
                    return fieldtools_1["default"].isEmpty(v.RATING_NEUTRAL_NUM) ? 0 : v.RATING_NEUTRAL_NUM;
                }
            },
            {
                th: function name(v) {
                    return '减持';
                },
                theadfix: 'jgtzpj',
                td: function (v) {
                    return fieldtools_1["default"].isEmpty(v.RATING_REDUCE_NUM) ? 0 : v.RATING_REDUCE_NUM;
                }
            },
            {
                th: function name(v) {
                    return '卖出';
                },
                theadfix: 'jgtzpj',
                td: function (v) {
                    return fieldtools_1["default"].isEmpty(v.RATING_SALE_NUM) ? 0 : v.RATING_SALE_NUM;
                }
            },
            {
                th: function name(v) {
                    var _a, _b, _c;
                    var year = (_c = (_b = (_a = v.data) === null || _a === void 0 ? void 0 : _a.data) === null || _b === void 0 ? void 0 : _b[0]) === null || _c === void 0 ? void 0 : _c.YEAR1;
                    return (year || '') + '每股收益';
                },
                td: function (v) {
                    return fieldtools_1["default"].fieldfmt(v.EPS1, { num: 3 });
                },
                icons: {
                    name: 'faq',
                    title: "\u9884\u6D4B\u6570\u636E\u6839\u636E\u5404\u673A\u6784\u53D1\u5E03\u7684\u7814\u62A5\u62A5\u544A\u6458\u5F55\u6240\u5F97\uFF0C\u4E0E\u672C\u7F51\u7AD9\u7ACB\u573A\u65E0\u5173\uFF1B\u5B9E\u9645\u6BCF\u80A1\u6536\u76CA\u4E3A\u7ECF\u6700\u65B0\u603B\u80A1\u672C\u8BA1\u7B97\u6240\u5F97"
                }
            },
            {
                th: function name(v) {
                    var _a, _b, _c;
                    var year = (_c = (_b = (_a = v.data) === null || _a === void 0 ? void 0 : _a.data) === null || _b === void 0 ? void 0 : _b[0]) === null || _c === void 0 ? void 0 : _c.YEAR2;
                    return (year || '') + '预测每股收益';
                },
                td: function (v) {
                    return fieldtools_1["default"].fieldfmt(v.EPS2, { num: 3 });
                }
            },
            {
                th: function name(v) {
                    var _a, _b, _c;
                    var year = (_c = (_b = (_a = v.data) === null || _a === void 0 ? void 0 : _a.data) === null || _b === void 0 ? void 0 : _b[0]) === null || _c === void 0 ? void 0 : _c.YEAR3;
                    return (year || '') + '预测每股收益';
                },
                td: function (v) {
                    return fieldtools_1["default"].fieldfmt(v.EPS3, { num: 3 });
                }
            },
            {
                th: function name(v) {
                    var _a, _b, _c;
                    var year = (_c = (_b = (_a = v.data) === null || _a === void 0 ? void 0 : _a.data) === null || _b === void 0 ? void 0 : _b[0]) === null || _c === void 0 ? void 0 : _c.YEAR4;
                    return (year || '') + '预测每股收益';
                },
                td: function (v) {
                    return fieldtools_1["default"].fieldfmt(v.EPS4, { num: 3 });
                }
            }
        ]
    },
    "indexmacresearch": {
        config: [
            {
                width: 44,
                th: function name(v) {
                    return '序号';
                },
                td: function (v) {
                    return v.index + 1;
                }
            },
            {
                width: 324,
                th: function name(v) {
                    return '报告名称';
                },
                td: function (v) {
                    return "<a href=\"/report/zw_macresearch.jshtml?encodeUrl=" + v.encodeUrl + ("\">" + v.title + "</a>");
                }
            },
            {
                th: function name(v) {
                    return '作者';
                },
                td: function (v) {
                    var authorStr = authorFormat(v, 12);
                    return authorStr;
                }
            },
            {
                th: function name(v) {
                    return '机构名称';
                },
                td: function (v) {
                    return "<a href=\"/report/orgpublish.jshtml?orgcode=" + v.orgCode + ("\">" + (v.orgSName ? v.orgSName : v.orgName) + "</a>");
                }
            },
            {
                paramname: 'count',
                th: function name(v) {
                    return '近一月机构宏<br>观研报数量';
                },
                td: function (v) {
                    return "" + (v.count != undefined ? v.count : '');
                }
            },
            {
                paramname: 'publishDate',
                th: function name(v) {
                    return '日期';
                },
                // sort: {
                //   asc: {
                //     sortkey: 'date',
                //     value: 'asc'
                //   },
                //   desc: {
                //     sortkey: 'date',
                //     value: 'desc'
                //   },
                //   default: {
                //     sortkey: 'date',
                //     value: ''
                //   }
                // },
                td: function (v) {
                    var date = (v.publishDate).split(' ')[0];
                    return date;
                }
            }
        ]
    },
    "indexnewstock": {
        config: [
            {
                width: 44,
                th: function name(v) {
                    return '序号';
                },
                td: function (v) {
                    return v.index + 1;
                }
            },
            {
                th: function name(v) {
                    return '股票代码';
                },
                td: function (v) {
                    return "<a href=\"//quote.eastmoney.com/unify/r/" + fieldtools_1["default"].datatoquotecode(v.stockCode) + "\">" + v.stockCode + "</a>";
                }
            },
            {
                th: function name(v) {
                    return '股票简称';
                },
                td: function (v) {
                    return "<a href=\"/stockdata/" + v.stockCode + (".html\">" + fieldtools_1["default"].fmtsname(v.stockName) + "</a>");
                }
            },
            {
                width: 60,
                th: function name(v) {
                    return '相关';
                },
                td: function (v) {
                    return "<a href=\"/report/" + v.stockCode + ".html\" class=\"red\">\u8BE6\u7EC6</a>&nbsp;\n                  <a href=\"//guba.eastmoney.com/list," + v.stockCode + ".html\">\u80A1\u5427</a>";
                }
            },
            {
                width: 220,
                th: function name(v) {
                    return '报告名称';
                },
                td: function (v) {
                    var date = ((v.publishDate).split(' ')[0]).replace(/-/g, '');
                    return "<a href=\"/report/info/" + v.infoCode + (".html\">" + v.title + "</a>");
                    // return `<a href="/report/ReportRedirect.aspx?date=` + date + `&code=` + v.infoCode + `">${v.title}</a>`
                }
            },
            {
                paramname: 'publishdate',
                th: function name(v) {
                    return '发布日期';
                },
                // sort: {
                //   asc: {
                //     sortkey: 'publishdate',
                //     value: 'asc'
                //   },
                //   desc: {
                //     sortkey: 'publishdate',
                //     value: 'desc'
                //   },
                //   default: {
                //     sortkey: 'publishdate',
                //     value: ''
                //   }
                // },
                td: function (v) {
                    var publishDate = v.publishDate ? (v.publishDate).substring(5, 11) : '';
                    return publishDate;
                }
            },
            {
                paramname: 'purchasedate',
                th: function name(v) {
                    return '申购日期';
                },
                // sort: {
                //   asc: {
                //     sortkey: 'purchasedate',
                //     value: 'asc'
                //   },
                //   desc: {
                //     sortkey: 'purchasedate',
                //     value: 'desc'
                //   },
                //   default: {
                //     sortkey: 'purchasedate',
                //     value: ''
                //   }
                // },
                td: function (v) {
                    var newPurchaseDate = v.newPurchaseDate ? (v.newPurchaseDate).substring(5, 11) : '';
                    return newPurchaseDate;
                }
            },
            {
                paramname: 'listingdate',
                th: function name(v) {
                    return '上市日期';
                },
                // sort: {
                //   asc: {
                //     sortkey: 'listingdate',
                //     value: 'asc'
                //   },
                //   desc: {
                //     sortkey: 'listingdate',
                //     value: 'desc'
                //   },
                //   default: {
                //     sortkey: 'listingdate',
                //     value: ''
                //   }
                // },
                td: function (v) {
                    var newListingDate = v.newListingDate ? (v.newListingDate).substring(5, 11) : '';
                    return newListingDate;
                }
            },
            {
                paramname: 'newIssuePrice',
                th: function name(v) {
                    return '发行价';
                },
                // sort: {
                //   asc: {
                //     sortkey: 'newIssuePrice',
                //     value: 'asc'
                //   },
                //   desc: {
                //     sortkey: 'newIssuePrice',
                //     value: 'desc'
                //   },
                //   default: {
                //     sortkey: 'newIssuePrice',
                //     value: ''
                //   }
                // },
                td: function (v) {
                    return v.newIssuePrice != '' ? Number(v.newIssuePrice).toFixed(2) : '';
                }
            },
            {
                paramname: 'newPeIssueA',
                th: function name(v) {
                    return '发行市<br>盈率';
                },
                // sort: {
                //   asc: {
                //     sortkey: 'newPeIssueA',
                //     value: 'asc'
                //   },
                //   desc: {
                //     sortkey: 'newPeIssueA',
                //     value: 'desc'
                //   },
                //   default: {
                //     sortkey: 'newPeIssueA',
                //     value: ''
                //   }
                // },
                td: function (v) {
                    return v.newPeIssueA != '' ? Number(v.newPeIssueA).toFixed(2) : '';
                }
            },
            {
                paramname: 'zxj',
                th: function name(v) {
                    return '最新价';
                },
                // sort: {
                //   asc: {
                //     sortkey: 'zxj',
                //     value: 'asc'
                //   },
                //   desc: {
                //     sortkey: 'zxj',
                //     value: 'desc'
                //   },
                //   default: {
                //     sortkey: 'zxj',
                //     value: ''
                //   }
                // },
                td: function (v) {
                    return '<span class="gghqspan" data-hqtype="zxj" data-market=' + v.market + ' data-code="' + v.stockCode + '"></span>';
                    // return '<span class="gghqspan" data-hqtype="zxj" data-market="SHENZHEN" data-code="300059"></span>'
                }
            },
            {
                paramname: 'zxsyl',
                th: function name(v) {
                    return '最新市盈率';
                },
                // sort: {
                //   asc: {
                //     sortkey: 'zxsyl',
                //     value: 'asc'
                //   },
                //   desc: {
                //     sortkey: 'zxsyl',
                //     value: 'desc'
                //   },
                //   default: {
                //     sortkey: 'zxsyl',
                //     value: ''
                //   }
                // },
                td: function (v) {
                    return '<span class="gghqspan" data-hqtype="zxsyl" data-market=' + v.market + ' data-code="' + v.stockCode + '"></span>';
                }
            },
            // {
            //   paramname:'hysyl',
            //   th: function name(v: any) {
            //     return '行业市盈率'
            //   },
            //   td: function (v: any) {
            //     return '<span class="hqspan" data-hqtype="hysyl" data-code="' + v.indvInduCode + '"></span>'
            //   }
            // },
            {
                th: function name(v) {
                    return '机构';
                },
                td: function (v) {
                    return "<a href=\"/report/orgpublish.jshtml?orgcode=" + v.orgCode + ("\">" + (v.orgSName ? v.orgSName : v.orgName) + "</a>");
                }
            }
        ]
    },
    "indexstrategyreport": {
        config: [
            {
                width: 44,
                th: function name(v) {
                    return '序号';
                },
                td: function (v) {
                    return v.index + 1;
                }
            },
            {
                width: 280,
                th: function name(v) {
                    return '报告名称';
                },
                td: function (v) {
                    return "<a href=\"/report/zw_strategy.jshtml?encodeUrl=" + v.encodeUrl + ("\">" + v.title + "</a>");
                }
            },
            {
                width: 220,
                th: function name(v) {
                    return '作者';
                },
                td: function (v) {
                    var authorStr = authorFormat(v, 20);
                    return authorStr;
                }
            },
            {
                th: function name(v) {
                    return '机构名称';
                },
                td: function (v) {
                    return "<a href=\"/report/orgpublish.jshtml?orgcode=" + v.orgCode + ("\">" + (v.orgSName ? v.orgSName : v.orgName) + "</a>");
                }
            },
            {
                paramname: 'count',
                th: function name(v) {
                    return '近一月机构策<br>略报告数量';
                },
                td: function (v) {
                    return "" + (v.count != undefined ? v.count : '');
                }
            },
            {
                width: 65,
                th: function name(v) {
                    return '日期';
                },
                td: function (v) {
                    var date = (v.publishDate).split(' ')[0];
                    return date;
                }
            }
        ]
    },
    "indexbroker": {
        config: [
            {
                width: 44,
                th: function name(v) {
                    return '序号';
                },
                td: function (v) {
                    return v.index + 1;
                }
            },
            {
                width: 222,
                th: function name(v) {
                    return '报告名称';
                },
                td: function (v) {
                    return "<a href=\"/report/zw_brokerreport.jshtml?encodeUrl=" + v.encodeUrl + ("\">" + v.title + "</a>");
                }
            },
            {
                th: function name(v) {
                    return '作者';
                },
                td: function (v) {
                    var authorStr = authorFormat(v, 20);
                    return authorStr;
                }
            },
            {
                th: function name(v) {
                    return '机构名称';
                },
                td: function (v) {
                    return "<a href=\"/report/orgpublish.jshtml?orgcode=" + v.orgCode + ("\">" + (v.orgSName ? v.orgSName : v.orgName) + "</a>");
                }
            },
            {
                th: function name(v) {
                    return '日期';
                },
                td: function (v) {
                    var date = (v.publishDate).split(' ')[0];
                    return date;
                }
            }
        ]
    },
    "eastmoneyanalyst": {
        config: [
            {
                width: 44,
                th: function name(v) {
                    return '序号';
                },
                td: function (v) {
                    return v.index + 1;
                }
            },
            {
                width: 74,
                th: function name(v) {
                    return '分析师名称';
                },
                td: function (v) {
                    return "<a href=\"/invest/invest/" + v.ANALYST_CODE + ".html\">" + v.ANALYST_NAME + "</a>";
                }
            },
            {
                th: function name(v) {
                    return '分析师单位';
                },
                td: function (v) {
                    return v.ORG_NAME;
                }
            },
            {
                th: function name(v) {
                    return '最新指数';
                },
                td: function (v) {
                    return (v.INDEX_VALUE).toFixed(2);
                }
            },
            {
                width: 158,
                th: function name(v) {
                    return '图表';
                },
                td: function (v) {
                    return "<a href=\"/invest/invest/" + v.ANALYST_CODE + ".html\">\u67E5\u770B\u8BE5\u5206\u6790\u5E08\u6307\u6570</a>";
                }
            },
            {
                th: function name(v) {
                    var displayYear = '';
                    var date = new Date();
                    var currentYear = date.getFullYear();
                    var myMonth = date.getMonth() + 1;
                    if (myMonth < 3) {
                        displayYear = currentYear - 1;
                    }
                    else {
                        displayYear = currentYear;
                    }
                    return displayYear + '年收益<br>率<span class="icon icon_desc"></span>';
                },
                td: function (v) {
                    return (v.YEAR_YIELD).toFixed(2) + '%';
                }
            },
            {
                th: function name(v) {
                    return '3个月收益率';
                },
                td: function (v) {
                    return (v.YIELD_3).toFixed(2) + '%';
                }
            },
            {
                th: function name(v) {
                    return '6个月收益率';
                },
                td: function (v) {
                    return (v.YIELD_6).toFixed(2) + '%';
                }
            },
            {
                th: function name(v) {
                    return '成分股个数';
                },
                td: function (v) {
                    return v.SECURITY_COUNT;
                }
            },
            {
                width: 140,
                th: function name(v) {
                    var displayYear = '';
                    var date = new Date();
                    var currentYear = date.getFullYear();
                    var myMonth = date.getMonth() + 1;
                    if (myMonth < 3) {
                        displayYear = currentYear - 1;
                    }
                    else {
                        displayYear = currentYear;
                    }
                    return displayYear + '最新个股评级';
                },
                td: function (v) {
                    return "<a href=\"/invest/invest/" + v.ANALYST_CODE + (".html#new\">" + v.NEWEST_STOCK_RATING + "</a>");
                }
            }
        ]
    },
    "industry": {
        config: [
            {
                th: function name(v) {
                    return '序号';
                },
                td: function (v) {
                    return v.index + 1;
                }
            },
            {
                th: function name(v) {
                    return '行业名称';
                },
                td: function (v) {
                    var bkcode = v.industryCode ? v.industryCode > 999 ? 'bk' + v.industryCode : 'bk0' + v.industryCode : '';
                    return "<a href=\"//quote.eastmoney.com/center/boardlist.html#boards-" + bkcode + ("\">" + v.industryName + "</a>");
                }
            },
            {
                th: function name(v) {
                    return '涨跌幅';
                },
                td: function (v) {
                    return '<span class="hqspan" data-hqtype="zdf" data-code="' + v.industryCode + '"></span>';
                }
            },
            {
                width: 162,
                th: function name(v) {
                    return '相关';
                },
                td: function (v) {
                    var bkcode = v.industryCode ? v.industryCode > 999 ? 'bk' + v.industryCode : 'bk0' + v.industryCode : '';
                    return "<a href=\"/report/industry.jshtml?hyid=" + v.industryCode + "\" class=\"red\">\u8BE6\u7EC6</a>&nbsp;\n                  <a href=\"/bkzj/" + v.industryCode + ".html\">\u8D44\u91D1\u6D41</a>&nbsp;\n                  <a href=\"//guba.eastmoney.com/list," + bkcode + ".html\">\u80A1\u5427</a>&nbsp;\n                  <a href=\"//stock.eastmoney.com/hangye/hy" + v.industryCode + ".html\">\u4E13\u533A</a>";
                }
            },
            {
                width: 222,
                th: function name(v) {
                    return '报告名称';
                },
                td: function (v) {
                    return "<a href=\"/report/zw_industry.jshtml?infocode=" + v.infoCode + ("\">" + v.title + "</a>");
                }
            },
            {
                th: function name(v) {
                    return '东财评级';
                },
                td: function (v) {
                    var emRatingName = v.emRatingName ? v.emRatingName : '-';
                    return "" + emRatingName;
                },
                icons: {
                    name: 'faq',
                    title: "\u6839\u636E\u673A\u6784\u62AB\u9732\u7684\u8BC4\u7EA7\u8BF4\u660E\u5212\u5206\u4E1C\u8D22\u8BC4\u7EA7\u6807\u51C6"
                }
            },
            {
                th: function name(v) {
                    return '评级<br>变动';
                },
                td: function (v) {
                    var ratingChangeName = getRatingChangeStr(v.ratingChange);
                    return "" + ratingChangeName;
                }
            },
            {
                th: function name(v) {
                    return '机构名称';
                },
                td: function (v) {
                    return "<a href=\"/report/orgpublish.jshtml?orgcode=" + v.orgCode + ("\">" + (v.orgSName ? v.orgSName : v.orgName) + "</a>");
                }
            },
            {
                paramname: 'count',
                th: function name(v) {
                    return '近一月行<br>业研报数';
                },
                td: function (v) {
                    return "" + v.count;
                }
            },
            {
                width: 65,
                paramname: 'publishDate',
                th: function name(v) {
                    return '日期';
                },
                sort: {
                    asc: {
                        sortkey: 'sort',
                        value: 'publishDate,asc'
                    },
                    desc: {
                        sortkey: 'sort',
                        value: 'publishDate,desc'
                    },
                    "default": {
                        sortkey: 'sort',
                        value: 'publishDate,default'
                    }
                },
                td: function (v) {
                    var date = (v.publishDate).split(' ')[0];
                    return "" + date;
                }
            }
        ]
    },
    "stock": {
        theadfix: {
            thisyear_ylyc: function (v) {
                var currentYear = v.currentYear ? v.currentYear : '';
                return currentYear + '盈利预测';
            },
            nextyear_ylyc: function (v) {
                var nextyear = v.currentYear ? v.currentYear + 1 : '';
                return nextyear + '盈利预测';
            }
        },
        config: [
            {
                th: function name(v) {
                    return '序号';
                },
                td: function (v) {
                    return v.index + 1;
                }
            },
            {
                width: 60,
                paramname: 'stockCode',
                th: function name(v) {
                    return '股票代码';
                },
                sort: {
                    asc: {
                        sortkey: 'sort',
                        value: 'stockCode,asc'
                    },
                    desc: {
                        sortkey: 'sort',
                        value: 'stockCode,desc'
                    },
                    "default": {
                        sortkey: 'sort',
                        value: 'stockCode,default'
                    }
                },
                td: function (v) {
                    return "<a href=\"//quote.eastmoney.com/unify/r/" + fieldtools_1["default"].datatoquotecode(v.stockCode) + "\">" + v.stockCode + "</a>";
                }
            },
            {
                width: 60,
                th: function name(v) {
                    return '股票简称';
                },
                td: function (v) {
                    return "<a href=\"/stockdata/" + v.stockCode + (".html\">" + fieldtools_1["default"].fmtsname(v.stockName) + "</a>");
                }
            },
            {
                width: 60,
                th: function name(v) {
                    return '相关';
                },
                td: function (v) {
                    return "<a href=\"/report/" + v.stockCode + ".html\" class=\"red\">\u8BE6\u7EC6</a>&nbsp;\n                  <a href=\"//guba.eastmoney.com/list," + v.stockCode + ".html\">\u80A1\u5427</a>";
                }
            },
            {
                width: 150,
                th: function name(v) {
                    return '报告名称';
                },
                td: function (v) {
                    var date = ((v.publishDate).split(' ')[0]).replace(/-/g, '');
                    return "<a href=\"/report/info/" + v.infoCode + (".html\">" + v.title + "</a>");
                    // return `<a href="/report/ReportRedirect.aspx?date=` + date + `&code=` + v.infoCode + `">${v.title}</a>`
                }
            },
            {
                th: function name(v) {
                    return '东财<br>评级';
                },
                td: function (v) {
                    var emRatingName = v.emRatingName ? v.emRatingName : '-';
                    return "" + emRatingName;
                },
                icons: {
                    name: 'faq',
                    title: "\u6839\u636E\u673A\u6784\u62AB\u9732\u7684\u8BC4\u7EA7\u8BF4\u660E\u5212\u5206\u4E1C\u8D22\u8BC4\u7EA7\u6807\u51C6"
                }
            },
            {
                th: function name(v) {
                    return '评级<br>变动';
                },
                td: function (v) {
                    var ratingChangeName = getRatingChangeStr(v.ratingChange);
                    return "" + ratingChangeName;
                }
            },
            {
                width: 70,
                th: function name(v) {
                    return '机构';
                },
                td: function (v) {
                    return "<a href=\"/report/orgpublish.jshtml?orgcode=" + v.orgCode + ("\">" + (v.orgSName ? v.orgSName : v.orgName) + "</a>");
                }
            },
            {
                paramname: 'count',
                th: function name(v) {
                    return '近一月<br>个股研<br>报数';
                },
                td: function (v) {
                    return "" + v.count;
                }
            },
            {
                paramname: 'predictThisYearEps',
                th: function (v) {
                    return '收益';
                },
                theadfix: "thisyear_ylyc",
                sort: {
                    asc: {
                        sortkey: 'sort',
                        value: 'predictThisYearEps,asc'
                    },
                    desc: {
                        sortkey: 'sort',
                        value: 'predictThisYearEps,desc'
                    },
                    "default": {
                        sortkey: 'sort',
                        value: 'predictThisYearEps,default'
                    }
                },
                td: function (v) {
                    var predictThisYearEps = "";
                    if (v.predictThisYearEps != undefined) {
                        predictThisYearEps = v.predictThisYearEps == '' ? "" : Number(v.predictThisYearEps).toFixed(3);
                    }
                    return predictThisYearEps;
                }
            },
            {
                paramname: 'predictThisYearPe',
                th: function name(v) {
                    return '市盈率';
                },
                theadfix: "thisyear_ylyc",
                sort: {
                    asc: {
                        sortkey: 'sort',
                        value: 'predictThisYearPe,asc'
                    },
                    desc: {
                        sortkey: 'sort',
                        value: 'predictThisYearPe,desc'
                    },
                    "default": {
                        sortkey: 'sort',
                        value: 'predictThisYearPe,default'
                    }
                },
                td: function (v) {
                    var predictThisYearPe = "";
                    if (v.predictThisYearPe != undefined) {
                        predictThisYearPe = v.predictThisYearPe == '' ? "" : Number(v.predictThisYearPe).toFixed(2);
                    }
                    return predictThisYearPe;
                }
            },
            {
                paramname: 'predictNextYearEps',
                th: function name(v) {
                    return '收益';
                },
                theadfix: "nextyear_ylyc",
                sort: {
                    asc: {
                        sortkey: 'sort',
                        value: 'predictNextYearEps,asc'
                    },
                    desc: {
                        sortkey: 'sort',
                        value: 'predictNextYearEps,desc'
                    },
                    "default": {
                        sortkey: 'sort',
                        value: 'predictNextYearEps,default'
                    }
                },
                td: function (v) {
                    var predictNextYearEps = "";
                    if (v.predictNextYearEps != undefined) {
                        predictNextYearEps = v.predictNextYearEps == '' ? "" : Number(v.predictNextYearEps).toFixed(3);
                    }
                    return predictNextYearEps;
                }
            },
            {
                paramname: 'predictNextYearPe',
                th: function name(v) {
                    return '市盈率';
                },
                theadfix: "nextyear_ylyc",
                sort: {
                    asc: {
                        sortkey: 'sort',
                        value: 'predictNextYearPe,asc'
                    },
                    desc: {
                        sortkey: 'sort',
                        value: 'predictNextYearPe,desc'
                    },
                    "default": {
                        sortkey: 'sort',
                        value: 'predictNextYearPe'
                    }
                },
                td: function (v) {
                    var predictNextYearPe = "";
                    if (v.predictNextYearPe != undefined) {
                        predictNextYearPe = v.predictNextYearPe == '' ? "" : Number(v.predictNextYearPe).toFixed(2);
                    }
                    return predictNextYearPe;
                }
            },
            {
                width: 60,
                th: function name(v) {
                    return '行业';
                },
                td: function (v) {
                    return "<a href=\"/report/industry.jshtml?hyid=" + v.indvInduCode + ("\">" + v.indvInduName + "</a>");
                }
            },
            {
                width: 65,
                paramname: 'publishDate',
                th: function name(v) {
                    return '日期';
                },
                sort: {
                    asc: {
                        sortkey: 'sort',
                        value: 'publishDate,asc'
                    },
                    desc: {
                        sortkey: 'sort',
                        value: 'publishDate,desc'
                    },
                    "default": {
                        sortkey: 'sort',
                        value: 'publishDate,default'
                    }
                },
                td: function (v) {
                    var publishDate = v.publishDate ? (v.publishDate).substring(0, 10) : '';
                    return "" + publishDate;
                }
            }
        ]
    },
    "profitforecast": {
        theadfix: {
            jgtzpj: function (v) {
                return '机构投资评级(近六个月)';
            }
        },
        config: [
            {
                width: 44,
                th: function name(v) {
                    return '序号';
                },
                td: function (v, p) {
                    return v.index + 1 + (p - 1) * 50;
                }
            },
            {
                paramname: 'stockCode',
                th: function name(v) {
                    return '代码';
                },
                td: function (v) {
                    return "<a href=\"//quote.eastmoney.com/unify/r/" + fieldtools_1["default"].datatoquotecode(v) + "\">" + v.SECURITY_CODE + "</a>";
                }
            },
            {
                th: function name(v) {
                    return '名称';
                },
                td: function (v) {
                    return "<a href=\"/stockdata/" + v.SECURITY_CODE + (".html\">" + fieldtools_1["default"].fmtsname(v.SECURITY_NAME_ABBR) + "</a>");
                }
            },
            {
                th: function name(v) {
                    return '相关';
                },
                td: function (v) {
                    return "<a href=\"/report/" + v.SECURITY_CODE + ".html\" class=\"red\">\u8BE6\u7EC6</a>&nbsp;\n                  <a href=\"//guba.eastmoney.com/list," + v.SECURITY_CODE + ".html\">\u80A1\u5427</a>";
                }
            },
            {
                paramname: 'total',
                th: function name(v) {
                    return '研报数';
                },
                td: function (v) {
                    return fieldtools_1["default"].isEmpty(v.RATING_ORG_NUM) ? 0 : v.RATING_ORG_NUM;
                }
            },
            {
                th: function name(v) {
                    return '买入';
                },
                theadfix: 'jgtzpj',
                td: function (v) {
                    return fieldtools_1["default"].isEmpty(v.RATING_BUY_NUM) ? 0 : v.RATING_BUY_NUM;
                }
            },
            {
                th: function name(v) {
                    return '增持';
                },
                theadfix: 'jgtzpj',
                td: function (v) {
                    return fieldtools_1["default"].isEmpty(v.RATING_ADD_NUM) ? 0 : v.RATING_ADD_NUM;
                }
            },
            {
                th: function name(v) {
                    return '中性';
                },
                theadfix: 'jgtzpj',
                td: function (v) {
                    return fieldtools_1["default"].isEmpty(v.RATING_NEUTRAL_NUM) ? 0 : v.RATING_NEUTRAL_NUM;
                }
            },
            {
                th: function name(v) {
                    return '减持';
                },
                theadfix: 'jgtzpj',
                td: function (v) {
                    return fieldtools_1["default"].isEmpty(v.RATING_REDUCE_NUM) ? 0 : v.RATING_REDUCE_NUM;
                }
            },
            {
                th: function name(v) {
                    return '卖出';
                },
                theadfix: 'jgtzpj',
                td: function (v) {
                    return fieldtools_1["default"].isEmpty(v.RATING_SALE_NUM) ? 0 : v.RATING_SALE_NUM;
                }
            },
            {
                th: function name(v) {
                    var _a, _b, _c;
                    var year = (_c = (_b = (_a = v.data) === null || _a === void 0 ? void 0 : _a.data) === null || _b === void 0 ? void 0 : _b[0]) === null || _c === void 0 ? void 0 : _c.YEAR1;
                    return (year || '') + '每股收益';
                },
                td: function (v) {
                    return fieldtools_1["default"].fieldfmt(v.EPS1, { num: 3 });
                },
                icons: {
                    name: 'faq',
                    title: "\u9884\u6D4B\u6570\u636E\u6839\u636E\u5404\u673A\u6784\u53D1\u5E03\u7684\u7814\u62A5\u62A5\u544A\u6458\u5F55\u6240\u5F97\uFF0C\u4E0E\u672C\u7F51\u7AD9\u7ACB\u573A\u65E0\u5173\uFF1B\u5B9E\u9645\u6BCF\u80A1\u6536\u76CA\u4E3A\u7ECF\u6700\u65B0\u603B\u80A1\u672C\u8BA1\u7B97\u6240\u5F97"
                }
            },
            {
                th: function name(v) {
                    var _a, _b, _c;
                    var year = (_c = (_b = (_a = v.data) === null || _a === void 0 ? void 0 : _a.data) === null || _b === void 0 ? void 0 : _b[0]) === null || _c === void 0 ? void 0 : _c.YEAR2;
                    return (year || '') + '预测每股收益';
                },
                td: function (v) {
                    return fieldtools_1["default"].fieldfmt(v.EPS2, { num: 3 });
                }
            },
            {
                th: function name(v) {
                    var _a, _b, _c;
                    var year = (_c = (_b = (_a = v.data) === null || _a === void 0 ? void 0 : _a.data) === null || _b === void 0 ? void 0 : _b[0]) === null || _c === void 0 ? void 0 : _c.YEAR3;
                    return (year || '') + '预测每股收益';
                },
                td: function (v) {
                    return fieldtools_1["default"].fieldfmt(v.EPS3, { num: 3 });
                }
            },
            {
                th: function name(v) {
                    var _a, _b, _c;
                    var year = (_c = (_b = (_a = v.data) === null || _a === void 0 ? void 0 : _a.data) === null || _b === void 0 ? void 0 : _b[0]) === null || _c === void 0 ? void 0 : _c.YEAR4;
                    return (year || '') + '预测每股收益';
                },
                td: function (v) {
                    return fieldtools_1["default"].fieldfmt(v.EPS4, { num: 3 });
                }
            }
        ]
    },
    "newstock": {
        config: [
            {
                th: function name(v) {
                    return '序号';
                },
                td: function (v) {
                    return v.index + 1;
                }
            },
            {
                width: 60,
                th: function name(v) {
                    return '股票代码';
                },
                td: function (v) {
                    return "<a href=\"//quote.eastmoney.com/unify/r/" + fieldtools_1["default"].datatoquotecode(v.stockCode) + "\">" + v.stockCode + "</a>";
                }
            },
            {
                width: 60,
                th: function name(v) {
                    return '股票简称';
                },
                td: function (v) {
                    return "<a href=\"/stockdata/" + v.stockCode + (".html\">" + fieldtools_1["default"].fmtsname(v.stockName) + "</a>");
                }
            },
            {
                width: 60,
                th: function name(v) {
                    return '相关';
                },
                td: function (v) {
                    return "<a href=\"/report/" + v.stockCode + ".html\" class=\"red\">\u8BE6\u7EC6</a>&nbsp;\n                  <a href=\"//guba.eastmoney.com/list," + v.stockCode + ".html\">\u80A1\u5427</a>";
                }
            },
            {
                width: 180,
                th: function name(v) {
                    return '报告名称';
                },
                td: function (v) {
                    return "<a href=\"/report/info/" + v.infoCode + (".html\">" + v.title + "</a>");
                }
            },
            {
                width: 54,
                paramname: 'publishDate',
                th: function name(v) {
                    return '发布日期';
                },
                sort: {
                    asc: {
                        sortkey: 'sort',
                        value: 'publishDate,asc'
                    },
                    desc: {
                        sortkey: 'sort',
                        value: 'publishDate,desc'
                    },
                    "default": {
                        sortkey: 'sort',
                        value: 'publishDate,default'
                    }
                },
                td: function (v) {
                    var publishDate = v.publishDate ? (v.publishDate).substring(5, 11) : '';
                    return publishDate;
                }
            },
            {
                width: 54,
                paramname: 'newPurchaseDate',
                th: function name(v) {
                    return '申购日期';
                },
                sort: {
                    asc: {
                        sortkey: 'sort',
                        value: 'newPurchaseDate,asc'
                    },
                    desc: {
                        sortkey: 'sort',
                        value: 'newPurchaseDate,desc'
                    },
                    "default": {
                        sortkey: 'sort',
                        value: 'newPurchaseDate,default'
                    }
                },
                td: function (v) {
                    var newPurchaseDate = v.newPurchaseDate ? (v.newPurchaseDate).substring(5, 11) : '';
                    return newPurchaseDate;
                }
            },
            {
                width: 54,
                paramname: 'newListingDate',
                th: function name(v) {
                    return '上市日期';
                },
                sort: {
                    asc: {
                        sortkey: 'sort',
                        value: 'newListingDate,asc'
                    },
                    desc: {
                        sortkey: 'sort',
                        value: 'newListingDate,desc'
                    },
                    "default": {
                        sortkey: 'sort',
                        value: 'newListingDate,default'
                    }
                },
                td: function (v) {
                    var newListingDate = v.newListingDate ? (v.newListingDate).substring(5, 11) : '';
                    return newListingDate;
                }
            },
            {
                width: 44,
                paramname: 'newIssuePrice',
                th: function name(v) {
                    return '发行价';
                },
                sort: {
                    asc: {
                        sortkey: 'sort',
                        value: 'newIssuePrice,asc'
                    },
                    desc: {
                        sortkey: 'sort',
                        value: 'newIssuePrice,desc'
                    },
                    "default": {
                        sortkey: 'sort',
                        value: 'newIssuePrice,default'
                    }
                },
                td: function (v) {
                    return v.newIssuePrice != '' ? Number(v.newIssuePrice).toFixed(2) : '';
                }
            },
            {
                paramname: 'newPeIssueA',
                th: function name(v) {
                    return '发行市<br>盈率';
                },
                sort: {
                    asc: {
                        sortkey: 'sort',
                        value: 'newPeIssueA,asc'
                    },
                    desc: {
                        sortkey: 'sort',
                        value: 'newPeIssueA,desc'
                    },
                    "default": {
                        sortkey: 'sort',
                        value: 'newPeIssueA,default'
                    }
                },
                td: function (v) {
                    return v.newPeIssueA != '' ? Number(v.newPeIssueA).toFixed(2) : '';
                }
            },
            {
                th: function name(v) {
                    return '最新价';
                },
                td: function (v) {
                    return '<span class="gghqspan" data-hqtype="zxj" data-market=' + v.market + ' data-code="' + v.stockCode + '"></span>';
                }
            },
            {
                th: function name(v) {
                    return '最新市<br>盈率';
                },
                td: function (v) {
                    return '<span class="gghqspan" data-hqtype="zxsyl" data-market=' + v.market + ' data-code="' + v.stockCode + '"></span>';
                }
            },
            {
                width: 70,
                th: function name(v) {
                    return '机构';
                },
                td: function (v) {
                    return "<a href=\"/report/orgpublish.jshtml?orgcode=" + v.orgCode + ("\">" + (v.orgSName ? v.orgSName : v.orgName) + "</a>");
                }
            }
        ]
    },
    "macresearch": {
        config: [
            {
                th: function name(v) {
                    return '序号';
                },
                td: function (v) {
                    return v.index + 1;
                }
            },
            {
                width: 288,
                th: function name(v) {
                    return '报告名称';
                },
                td: function (v) {
                    return "<a href=\"/report/zw_macresearch.jshtml?encodeUrl=" + v.encodeUrl + ("\">" + v.title + "</a>");
                }
            },
            {
                width: 210,
                th: function name(v) {
                    return '作者';
                },
                td: function (v) {
                    var authorStr = authorFormat(v);
                    return authorStr;
                }
            },
            {
                width: 120,
                th: function name(v) {
                    return '机构名称';
                },
                td: function (v) {
                    return "<a href=\"/report/orgpublish.jshtml?orgcode=" + v.orgCode + ("\">" + (v.orgSName ? v.orgSName : v.orgName) + "</a>");
                }
            },
            {
                paramname: 'count',
                th: function name(v) {
                    return '近一月机构宏<br>观研报数量';
                },
                td: function (v) {
                    return "" + (v.count != undefined ? v.count : '');
                }
            },
            {
                paramname: 'publishDate',
                th: function name(v) {
                    return '日期';
                },
                sort: {
                    asc: {
                        sortkey: 'sort',
                        value: 'publishDate,asc'
                    },
                    desc: {
                        sortkey: 'sort',
                        value: 'publishDate,desc'
                    },
                    "default": {
                        sortkey: 'sort',
                        value: 'publishDate,default'
                    }
                },
                td: function (v) {
                    var publishDate = v.publishDate ? (v.publishDate).substring(0, 11) : '';
                    return publishDate;
                }
            }
        ]
    },
    "strategyreport": {
        config: [
            {
                width: 60,
                th: function name(v) {
                    return '序号';
                },
                td: function (v) {
                    return v.index + 1;
                }
            },
            {
                width: 288,
                th: function name(v) {
                    return '报告名称';
                },
                td: function (v) {
                    return "<a href=\"/report/zw_strategy.jshtml?encodeUrl=" + v.encodeUrl + ("\">" + v.title + "</a>");
                }
            },
            {
                width: 180,
                th: function name(v) {
                    return '作者';
                },
                td: function (v) {
                    var authorStr = authorFormat(v);
                    return authorStr;
                }
            },
            {
                width: 90,
                th: function name(v) {
                    return '机构名称';
                },
                td: function (v) {
                    return "<a href=\"/report/orgpublish.jshtml?orgcode=" + v.orgCode + ("\">" + (v.orgSName ? v.orgSName : v.orgName) + "</a>");
                }
            },
            {
                width: 90,
                paramname: 'count',
                th: function name(v) {
                    return '近一月机构策略报告数量';
                },
                td: function (v) {
                    return "" + (v.count != undefined ? v.count : '');
                }
            },
            {
                width: 90,
                paramname: 'publishDate',
                th: function name(v) {
                    return '日期';
                },
                sort: {
                    asc: {
                        sortkey: 'sort',
                        value: 'publishDate,asc'
                    },
                    desc: {
                        sortkey: 'sort',
                        value: 'publishDate,desc'
                    },
                    "default": {
                        sortkey: 'sort',
                        value: 'publishDate,default'
                    }
                },
                td: function (v) {
                    var publishDate = v.publishDate ? (v.publishDate).substring(0, 11) : '';
                    return publishDate;
                }
            }
        ]
    },
    "brokerreport": {
        config: [
            {
                th: function name(v) {
                    return '序号';
                },
                td: function (v) {
                    return v.index + 1;
                }
            },
            {
                width: 268,
                th: function name(v) {
                    return '报告名称';
                },
                td: function (v) {
                    return "<a href=\"/report/zw_brokerreport.jshtml?encodeUrl=" + v.encodeUrl + ("\">" + v.title + "</a>");
                }
            },
            {
                width: 220,
                th: function name(v) {
                    return '作者';
                },
                td: function (v) {
                    var authorStr = authorFormat(v);
                    return authorStr;
                }
            },
            {
                width: 220,
                th: function name(v) {
                    return '机构名称';
                },
                td: function (v) {
                    return "<a href=\"/report/orgpublish.jshtml?orgcode=" + v.orgCode + ("\">" + (v.orgSName ? v.orgSName : v.orgName) + "</a>");
                }
            },
            {
                paramname: 'publishDate',
                th: function name(v) {
                    return '日期';
                },
                sort: {
                    asc: {
                        sortkey: 'sort',
                        value: 'publishDate,asc'
                    },
                    desc: {
                        sortkey: 'sort',
                        value: 'publishDate,desc'
                    },
                    "default": {
                        sortkey: 'sort',
                        value: 'publishDate,default'
                    }
                },
                td: function (v) {
                    var publishDate = v.publishDate ? (v.publishDate).substring(0, 11) : '';
                    return publishDate;
                }
            }
        ]
    },
    "orgpublish": {
        config: [
            {
                th: function name(v) {
                    return '序号';
                },
                td: function (v) {
                    return v.index + 1;
                }
            },
            {
                width: 282,
                th: function name(v) {
                    return '报告名称';
                },
                td: function (v) {
                    //return `<a>${v.title}</a>`
                    return getZWUrl(v);
                }
            },
            {
                th: function name(v) {
                    return '报告类型';
                },
                td: function (v) {
                    return geTypeUrl(v);
                }
            },
            {
                th: function name(v) {
                    return '研究对象';
                },
                td: function (v) {
                    return getYJDXUrl(v);
                }
            },
            {
                width: 180,
                th: function name(v) {
                    return '作者';
                },
                td: function (v) {
                    var authorStr = authorFormat(v);
                    return authorStr;
                }
            },
            {
                th: function name(v) {
                    return '机构';
                },
                td: function (v) {
                    return "<a href=\"/report/orgpublish.jshtml?orgcode=" + v.orgCode + ("\">" + (v.orgSName ? v.orgSName : v.orgName) + "</a>");
                }
            },
            {
                paramname: 'publishDate',
                th: function name(v) {
                    return '日期';
                },
                sort: {
                    asc: {
                        sortkey: 'sort',
                        value: 'publishDate,asc'
                    },
                    desc: {
                        sortkey: 'sort',
                        value: 'publishDate,desc'
                    },
                    "default": {
                        sortkey: 'sort',
                        value: 'publishDate,default'
                    }
                },
                td: function (v) {
                    var publishDate = v.publishDate ? (v.publishDate).split(' ')[0] : '';
                    return publishDate;
                }
            }
        ]
    },
    "personalpublish": {
        config: [
            {
                th: function name(v) {
                    return '序号';
                },
                td: function (v) {
                    return v.index + 1;
                }
            },
            {
                width: 282,
                th: function name(v) {
                    return '报告名称';
                },
                td: function (v) {
                    return getZWUrl(v);
                }
            },
            {
                th: function name(v) {
                    return '报告类型';
                },
                td: function (v) {
                    return geTypeUrl(v);
                }
            },
            {
                th: function name(v) {
                    return '研究对象';
                },
                td: function (v) {
                    return getYJDXUrl(v);
                }
            },
            {
                width: 180,
                th: function name(v) {
                    return '作者';
                },
                td: function (v) {
                    var authorStr = authorFormat(v);
                    return authorStr;
                }
            },
            {
                th: function name(v) {
                    return '机构';
                },
                td: function (v) {
                    return "<a href=\"/report/orgpublish.jshtml?orgcode=" + v.orgCode + ("\">" + (v.orgSName ? v.orgSName : v.orgName) + "</a>");
                }
            },
            {
                paramname: 'publishDate',
                th: function name(v) {
                    return '日期';
                },
                sort: {
                    asc: {
                        sortkey: 'sort',
                        value: 'publishDate,asc'
                    },
                    desc: {
                        sortkey: 'sort',
                        value: 'publishDate,desc'
                    },
                    "default": {
                        sortkey: 'sort',
                        value: 'publishDate,default'
                    }
                },
                td: function (v) {
                    var publishDate = v.publishDate ? (v.publishDate).split(' ')[0] : '';
                    return publishDate;
                }
            }
        ]
    },
    "singlestock": {
        config: [
            {
                th: function name(v) {
                    return '序号';
                },
                td: function (v) {
                    return v.index + 1;
                }
            },
            {
                width: 248,
                th: function name(v) {
                    return '报告名称';
                },
                td: function (v) {
                    return "<a href=\"/report/info/" + v.infoCode + (".html\">" + v.title + "</a>");
                }
            },
            {
                th: function name(v) {
                    return '东财<br>评级';
                },
                td: function (v) {
                    var emRatingName = v.emRatingName ? v.emRatingName : '-';
                    return "" + emRatingName;
                },
                icons: {
                    name: 'faq',
                    title: "\u6839\u636E\u673A\u6784\u62AB\u9732\u7684\u8BC4\u7EA7\u8BF4\u660E\u5212\u5206\u4E1C\u8D22\u8BC4\u7EA7\u6807\u51C6"
                }
            },
            {
                th: function name(v) {
                    return '评级<br>变动';
                },
                td: function (v) {
                    var ratingChangeName = getRatingChangeStr(v.ratingChange);
                    return "" + ratingChangeName;
                }
            },
            {
                th: function name(v) {
                    return '作者';
                },
                td: function (v) {
                    var authorStr = authorFormat(v);
                    return authorStr;
                }
            },
            {
                th: function name(v) {
                    return '机构';
                },
                td: function (v) {
                    return "<a href=\"/report/orgpublish.jshtml?orgcode=" + v.orgCode + ("\">" + (v.orgSName ? v.orgSName : v.orgName) + "</a>");
                }
            },
            {
                paramname: 'publishDate',
                th: function name(v) {
                    return '日期';
                },
                sort: {
                    asc: {
                        sortkey: 'sort',
                        value: 'publishDate,asc'
                    },
                    desc: {
                        sortkey: 'sort',
                        value: 'publishDate,desc'
                    },
                    "default": {
                        sortkey: 'sort',
                        value: 'publishDate,default'
                    }
                },
                td: function (v) {
                    var publishDate = v.publishDate ? (v.publishDate).split(' ')[0] : '';
                    return publishDate;
                }
            }
        ]
    },
    //------------正文----------
    //策略报告正文：策略报告
    "zw_stratyge": {
        config: [
            {
                width: 312,
                th: function name(v) {
                    return '标题';
                },
                td: function (v) {
                    return "<a href=\"/report/zw_strategy.jshtml?encodeUrl=" + v.encodeUrl + ("\">" + v.title + "</a>");
                }
            },
            {
                th: function name(v) {
                    return '报告日期';
                },
                td: function (v) {
                    var publishDate = v.publishDate ? (v.publishDate).substring(0, 11) : '';
                    return publishDate;
                }
            },
            {
                th: function name(v) {
                    return '机构名称';
                },
                td: function (v) {
                    return "<a href=\"/report/orgpublish.jshtml?orgcode=" + v.orgCode + ("\">" + (v.orgSName ? v.orgSName : v.orgName) + "</a>");
                }
            },
            {
                width: 160,
                th: function name(v) {
                    return '研究员';
                },
                td: function (v) {
                    var arr = v.researcher.split(',');
                    if (arr.length < 4) {
                        return "<span>" + v.researcher + "</span>";
                    }
                    else {
                        return "<span title=\"" + arr.join(',') + "\">" + arr.slice(0, 3).join(',') + "...</span>";
                    }
                }
            }
        ]
    },
    //策略报告正文：zuixin
    "zw_neweast": {
        config: [
            {
                width: 312,
                th: function name(v) {
                    return '标题';
                },
                td: function (v) {
                    return "<a href=\"/report/zw_strategy.jshtml?encodeUrl=" + v.encodeUrl + ("\">" + v.title + "</a>");
                }
            },
            {
                th: function name(v) {
                    return '报告日期';
                },
                td: function (v) {
                    var publishDate = v.publishDate ? (v.publishDate).substring(0, 11) : '';
                    return publishDate;
                }
            },
            {
                th: function name(v) {
                    return '机构名称';
                },
                td: function (v) {
                    return "<a href=\"/report/orgpublish.jshtml?orgcode=" + v.orgCode + ("\">" + (v.orgSName ? v.orgSName : v.orgName) + "</a>");
                }
            },
            {
                width: 160,
                th: function name(v) {
                    return '研究员';
                },
                td: function (v) {
                    var arr = v.researcher.split(',');
                    if (arr.length < 4) {
                        return "<span>" + v.researcher + "</span>";
                    }
                    else {
                        return "<span title=\"" + arr.join(',') + "\">" + arr.slice(0, 3).join(',') + "...</span>";
                    }
                }
            }
        ]
    },
    //策略报告正文：热门行业追踪 
    "zw_hotindustry": {},
    //行业研报正文：文中涉及到的个股
    "zw_industry_stock": {
        config: [
            {
                th: function name(v) {
                    return '代码';
                },
                td: function (v) {
                    return "<a href=\"//quote.eastmoney.com/unify/r/" + fieldtools_1["default"].datatoquotecode(v.code) + "\">" + v.code + "</a>";
                }
            },
            {
                th: function name(v) {
                    return '名称';
                },
                td: function (v) {
                    return "<a href=\"/stockdata/" + v.code + (".html\">" + fieldtools_1["default"].fmtsname(v.name) + "</a>");
                }
            },
            {
                width: 120,
                th: function name(v) {
                    return '相关';
                },
                td: function (v) {
                    return "<a href=\"/report/" + v.code + ".html\">\u7814\u62A5</a>&nbsp;\n                  <a href=\"/zjlx/" + v.code + ".html\">\u8D44\u91D1\u6D41</a>&nbsp;\n                  <a href=\"//guba.eastmoney.com/list," + v.code + ".html\">\u80A1\u5427</a>";
                }
            },
            {
                th: function name(v) {
                    return '最新价';
                },
                td: function (v) {
                    var $span;
                    if (v.zde < 0) {
                        $span = "<span class='green'>" + v.zxj + "</span>";
                    }
                    else if (v.zde > 0) {
                        $span = "<span class='red'>" + v.zxj + "</span>";
                    }
                    else {
                        $span = "<span class=''>" + v.zxj + "</span>";
                    }
                    return $span;
                }
            },
            {
                th: function name(v) {
                    return '涨跌额';
                },
                td: function (v) {
                    var $span;
                    if (v.zde < 0) {
                        $span = "<span class='green'>" + v.zde + "</span>";
                    }
                    else if (v.zde > 0) {
                        $span = "<span class='red'>" + v.zde + "</span>";
                    }
                    else {
                        $span = "<span class=''>" + v.zde + "</span>";
                    }
                    return $span;
                }
            },
            {
                th: function name(v) {
                    return '涨跌幅';
                },
                td: function (v) {
                    var $span;
                    if (v.zde < 0) {
                        $span = "<span class='green'>" + v.zdf + "</span>";
                    }
                    else if (v.zde > 0) {
                        $span = "<span class='red'>" + v.zdf + "</span>";
                    }
                    else {
                        $span = "<span class=''>" + v.zdf + "</span>";
                    }
                    return $span;
                }
            }
        ]
    },
    //行业研报正文：所在行业最新研究报告
    "zw_indsneweast": {
        config: [
            {
                width: 312,
                th: function name(v) {
                    return '标题';
                },
                td: function (v) {
                    var title = v.title;
                    return "<a href=\"/report/zw_industry.jshtml?infocode=" + v.infoCode + ("\">" + v.title + "</a>");
                }
            },
            {
                th: function name(v) {
                    return '东财评级';
                },
                td: function (v) {
                    var emRatingName = v.emRatingName ? v.emRatingName : '';
                    return "" + emRatingName;
                },
                icons: {
                    name: 'faq',
                    title: "\u6839\u636E\u673A\u6784\u62AB\u9732\u7684\u8BC4\u7EA7\u8BF4\u660E\u5212\u5206\u4E1C\u8D22\u8BC4\u7EA7\u6807\u51C6"
                }
            },
            {
                th: function name(v) {
                    return '评级变动';
                },
                td: function (v) {
                    var ratingChangeName = getRatingChangeStr(v.ratingChange);
                    return "" + ratingChangeName;
                }
            },
            {
                th: function name(v) {
                    return '报告日期';
                },
                td: function (v) {
                    var publishDate = v.publishDate ? (v.publishDate).substring(0, 11) : '';
                    return publishDate;
                }
            },
            {
                th: function name(v) {
                    return '机构名称';
                },
                td: function (v) {
                    return "<a href=\"/report/orgpublish.jshtml?orgcode=" + v.orgCode + ("\">" + (v.orgSName ? v.orgSName : v.orgName) + "</a>");
                }
            }
        ]
    },
    //行业研报正文  ：热门行业评级一览
    "zw_indshotstock": {
        theadfix: {
            orgGrade_6: function (v) {
                return '机构投资评级（近六个月）';
            }
        },
        config: [
            {
                width: 60,
                th: function name(v) {
                    return '名称';
                },
                td: function (v) {
                    return "<a href=\"/stockdata/" + v.code + (".html\">" + fieldtools_1["default"].fmtsname(v.name) + "</a>");
                }
            },
            {
                width: 60,
                th: function name(v) {
                    return '涨跌幅';
                },
                td: function (v) {
                    var zdf = v.zdf;
                    var span = '';
                    // span = zdf == '-' ? span = zdf : String(zdf).indexOf('-')>-1 ? span = '<span class="green">' + zdf + '%</span>' : span = '<span class="red">' + zdf + '%</span>';
                    if (zdf > 0) {
                        span = '<span class="red">' + zdf + '%</span>';
                    }
                    else if (zdf < 0) {
                        span = '<span class="green">' + zdf + '%</span>';
                    }
                    else if (zdf == 0) {
                        span = '<span>0.00%</span>';
                    }
                    else if (zdf == '-') {
                        span = '<span>-</span>';
                    }
                    return span;
                    // return '<span class="gghqspan" data-hqtype="zdf" data-market="'+v.market+'" data-code="' + v.stockCode + '"></span>'
                }
            },
            {
                width: 40,
                th: function name(v) {
                    return '研报数';
                },
                td: function (v) {
                    // return !!v.total ? `<a href="">` + v.total+`</a>` :''
                    return v.total !== '' ? v.total : '';
                }
            },
            {
                th: function name(v) {
                    return '买入';
                },
                theadfix: 'orgGrade_6',
                td: function (v) {
                    return v.rateBuy;
                }
            },
            {
                th: function name(v) {
                    return '增持';
                },
                theadfix: 'orgGrade_6',
                td: function (v) {
                    return v.rateIncrease;
                }
            },
            {
                th: function name(v) {
                    return '中性';
                },
                theadfix: 'orgGrade_6',
                td: function (v) {
                    return v.rateNeutral;
                }
            },
            {
                th: function name(v) {
                    return '减持';
                },
                theadfix: 'orgGrade_6',
                td: function (v) {
                    return v.rateReduce;
                }
            },
            {
                th: function name(v) {
                    return '卖出';
                },
                theadfix: 'orgGrade_6',
                td: function (v) {
                    return v.rateSellout;
                }
            }
        ]
    },
    //行业研报正文  ：行业个股未来3年盈利预测
    "zw_indsstockforecast": {
        config: [
            {
                width: 60,
                th: function name(v) {
                    return '名称';
                },
                td: function (v) {
                    return "<a href=\"/stockdata/" + v.SECURITY_CODE + (".html\">" + fieldtools_1["default"].fmtsname(v.SECURITY_NAME_ABBR) + "</a>");
                }
            },
            {
                width: 70,
                th: function name(v) {
                    return '相关';
                },
                td: function (v) {
                    return "<a href=\"/report/" + v.SECURITY_CODE + ".html\" class=\"red\">\u8BE6\u7EC6</a>&nbsp;\n          <a href=\"//guba.eastmoney.com/list," + v.SECURITY_CODE + ".html\">\u80A1\u5427</a>";
                }
            },
            {
                th: function name(v) {
                    var _a, _b, _c;
                    var year = (_c = (_b = (_a = v.data) === null || _a === void 0 ? void 0 : _a.data) === null || _b === void 0 ? void 0 : _b[0]) === null || _c === void 0 ? void 0 : _c.YEAR2;
                    return (year || '') + '<br>预测收益';
                },
                // theadfix: 'thisyear_ycsy',
                td: function (v) {
                    return fieldtools_1["default"].fieldfmt(v.EPS2, { num: 3 });
                }
            },
            {
                th: function name(v) {
                    var _a, _b, _c;
                    var year = (_c = (_b = (_a = v.data) === null || _a === void 0 ? void 0 : _a.data) === null || _b === void 0 ? void 0 : _b[0]) === null || _c === void 0 ? void 0 : _c.YEAR3;
                    return (year || '') + '<br>预测收益';
                },
                // theadfix: 'nextyear_ycsy',
                td: function (v) {
                    return fieldtools_1["default"].fieldfmt(v.EPS3, { num: 3 });
                }
            },
            {
                th: function name(v) {
                    var _a, _b, _c;
                    var year = (_c = (_b = (_a = v.data) === null || _a === void 0 ? void 0 : _a.data) === null || _b === void 0 ? void 0 : _b[0]) === null || _c === void 0 ? void 0 : _c.YEAR4;
                    return (year || '') + '<br>预测收益';
                },
                // theadfix: 'afteryear_ycsy',
                td: function (v) {
                    return fieldtools_1["default"].fieldfmt(v.EPS4, { num: 3 });
                }
            }
        ]
    },
    //行业研报正文：热门行业追踪
    "zw_indshotindustry": {
        config: [
            {
                width: 80,
                th: function name(v) {
                    return '名称';
                },
                td: function (v) {
                    // return `<a>${v.name}</a>`   
                    return "<a href=\"//quote.eastmoney.com/center/boardlist.html#boards-" + v.code + ("\" title=\"" + v.name + "\">" + v.name + "</a>");
                    // return `<a href="//quote.eastmoney.com/center/list.html#` + v.stockcode+`">${v.name}</a>`
                }
            },
            {
                width: 40,
                th: function name(v) {
                    return '涨跌幅';
                },
                td: function (v) {
                    var zdf = v.zdf;
                    var span = '';
                    if (zdf > 0) {
                        span = '<span class="red">' + zdf + '%</span>';
                    }
                    else if (zdf < 0) {
                        span = '<span class="green">' + zdf + '%</span>';
                    }
                    else if (zdf == 0) {
                        span = '<span>0.00%</span>';
                    }
                    if (zdf == '-') {
                        span = zdf;
                    }
                    return span;
                }
            },
            {
                width: 70,
                th: function name(v) {
                    return '相关';
                },
                td: function (v) {
                    var bkcode = v.code;
                    bkcode = bkcode.charAt(2) == '0' ? bkcode.substring(3) : bkcode.substring(2);
                    return "<a href=\"/report/industry.jshtml?hyid=" + bkcode + "\">\u7814\u62A5</a>\n                  <a href=\"/bkzj/" + bkcode + ".html\">\u8D44\u91D1\u6D41</a>";
                }
            },
            {
                width: 60,
                th: function name(v) {
                    return '领涨股票';
                },
                td: function (v) {
                    return v.lzgp == '-' ? v.lzgp : "<a href=\"//quote.eastmoney.com/unify/r/" + fieldtools_1["default"].datatoquotecode(v.stockcode) + "\">" + v.lzgp + "</a>";
                }
            },
            {
                width: 70,
                th: function name(v) {
                    return '相关';
                },
                td: function (v) {
                    return "<a href=\"/report/" + v.stockcode + ".html\">\u7814\u62A5</a>\n                  <a href=\"/zjlx/" + v.stockcode + ".html\">\u8D44\u91D1\u6D41</a>";
                }
            }
        ]
    },
    //行业正文：个股财务指标 
    "zw_indggcw_table": {
        config: [
            {
                width: 30,
                th: function name(v) {
                    return '序号';
                },
                td: function (v) {
                    return v.index + 1;
                }
            },
            {
                width: 90,
                th: function name(v) {
                    return '名称';
                },
                td: function (v) {
                    return "<a href=\"/stockdata/" + v.code + (".html\">" + fieldtools_1["default"].fmtsname(v.name) + "</a>");
                }
            },
            {
                width: 90,
                th: function name(v) {
                    return '相关';
                },
                td: function (v) {
                    return "<a href=\"/report/" + v.code + ".html\">\u7814\u62A5</a>&nbsp;\n                  <a href=\"//guba.eastmoney.com/list," + v.code + ".html\">\u80A1\u5427</a>";
                }
            },
            {
                th: function name(v) {
                    var fid = v.data.data[0].fid;
                    var titlename = '';
                    if (fid == 'f112') {
                        titlename = '每股收益(元)';
                    }
                    else if (fid == 'f113') {
                        titlename += '每股净资产(元)';
                    }
                    else if (fid == 'f37') {
                        titlename = '净资产收益率(%)';
                    }
                    else if (fid == 'f48') {
                        titlename = '每股未分配利润(元)';
                    }
                    else if (fid == 'f45') {
                        titlename = '净利润(亿元)';
                    }
                    else if (fid == 'f40') {
                        titlename = '营业收入(亿元)';
                    }
                    else if (fid == 'f41') {
                        titlename = '营业收入同比(%)';
                    }
                    return titlename;
                    // return '<span>每股收益(元)</span>'
                },
                td: function (v) {
                    // let profit = v.value ? Number(v.value).toFixed(2):'';
                    return v.value;
                }
            }
        ]
    },
    //宏观研究正文 ：最新宏观研究
    "zw_macsearch": {
        config: [
            {
                width: 312,
                th: function name(v) {
                    return '标题';
                },
                td: function (v) {
                    return "<a href=\"/report/zw_macresearch.jshtml?encodeUrl=" + v.encodeUrl + ("\">" + v.title + "</a>");
                }
            },
            {
                th: function name(v) {
                    return '报告日期';
                },
                td: function (v) {
                    var publishDate = v.publishDate ? (v.publishDate).substring(0, 11) : '';
                    return publishDate;
                }
            },
            {
                th: function name(v) {
                    return '机构名称';
                },
                td: function (v) {
                    return "<a href=\"/report/orgpublish.jshtml?orgcode=" + v.orgCode + ("\">" + (v.orgSName ? v.orgSName : v.orgName) + "</a>");
                }
            },
            {
                width: 160,
                th: function name(v) {
                    return '研究员';
                },
                td: function (v) {
                    var arr = v.researcher.split(',');
                    if (arr.length < 4) {
                        return "<span>" + v.researcher + "</span>";
                    }
                    else {
                        return "<span title=\"" + arr.join(',') + "\">" + arr.slice(0, 3).join(',') + "...</span>";
                    }
                }
            }
        ]
    },
    //券商晨报正文：最新券商 
    "zw_broker": {
        config: [
            {
                width: 212,
                th: function name(v) {
                    return '标题';
                },
                td: function (v) {
                    return "<a href=\"/report/zw_brokerreport.jshtml?encodeUrl=" + v.encodeUrl + ("\">" + v.title + "</a>");
                }
            },
            {
                th: function name(v) {
                    return '报告日期';
                },
                td: function (v) {
                    var publishDate = v.publishDate ? (v.publishDate).substring(0, 11) : '';
                    return publishDate;
                }
            },
            {
                th: function name(v) {
                    return '机构名称';
                },
                td: function (v) {
                    return "<a href=\"/report/orgpublish.jshtml?orgcode=" + v.orgCode + ("\">" + (v.orgSName ? v.orgSName : v.orgName) + "</a>");
                }
            },
            {
                th: function name(v) {
                    return '研究员';
                },
                td: function (v) {
                    var arr = v.researcher.split(',');
                    if (arr.length < 4) {
                        return "<span>" + v.researcher + "</span>";
                    }
                    else {
                        return "<span title=\"" + arr.join(',') + "\">" + arr.slice(0, 3).join(',') + "...</span>";
                    }
                }
            }
        ]
    },
    //券商晨报正文：最新研究  暂定
    "zw_neweastbroker": {
        config: [
            {
                width: 312,
                th: function name(v) {
                    return '标题';
                },
                td: function (v) {
                    return "<a href=\"/report/zw_macresearch.jshtml?encodeUrl=" + v.encodeUrl + ("\">" + v.title + "</a>");
                }
            },
            {
                th: function name(v) {
                    return '报告日期';
                },
                td: function (v) {
                    var publishDate = v.publishDate ? (v.publishDate).substring(0, 11) : '';
                    return publishDate;
                }
            },
            {
                th: function name(v) {
                    return '机构名称';
                },
                td: function (v) {
                    return "<a href=\"/report/orgpublish.jshtml?orgcode=" + v.orgCode + ("\">" + (v.orgSName ? v.orgSName : v.orgName) + "</a>");
                }
            },
            {
                width: 160,
                th: function name(v) {
                    return '研究员';
                },
                td: function (v) {
                    var arr = v.researcher.split(',');
                    if (arr.length < 4) {
                        return "<span>" + v.researcher + "</span>";
                    }
                    else {
                        return "<span title=\"" + arr.join(',') + "\">" + arr.slice(0, 3).join(',') + "...</span>";
                    }
                }
            }
        ]
    },
    //单个个股正文行业最新
    "zw_indsneweast_stock": {
        config: [
            {
                width: 180,
                th: function name(v) {
                    return '标题';
                },
                td: function (v) {
                    var title = utils_1["default"].txtLeft(v.title, 24, true);
                    return "<a href=\"/report/zw_industry.jshtml?infocode=" + v.infoCode + ("\">" + title + "</a>");
                }
            },
            {
                th: function name(v) {
                    return '东财评级';
                },
                td: function (v) {
                    var emRatingName = v.emRatingName ? v.emRatingName : '';
                    return "" + emRatingName;
                },
                icons: {
                    name: 'faq',
                    title: "\u6839\u636E\u673A\u6784\u62AB\u9732\u7684\u8BC4\u7EA7\u8BF4\u660E\u5212\u5206\u4E1C\u8D22\u8BC4\u7EA7\u6807\u51C6"
                }
            },
            {
                th: function name(v) {
                    return '报告日期';
                },
                td: function (v) {
                    var publishDate = v.publishDate ? (v.publishDate).substring(5, 11) : '';
                    return publishDate;
                }
            }
        ]
    },
    "zw_orgneweast_stock": {
        config: [
            {
                width: 180,
                th: function name(v) {
                    return '标题';
                },
                td: function (v) {
                    var title = utils_1["default"].txtLeft(v.title, 24, true);
                    // return `<a href="zw_industry.jshtml?infocode=` + v.infoCode + `">${title}</a>`
                    return "<a href=\"/report/info/" + v.infoCode + (".html\">" + title + "</a>");
                }
            },
            {
                th: function name(v) {
                    return '东财评级';
                },
                td: function (v) {
                    var emRatingName = v.emRatingName ? v.emRatingName : '';
                    return "" + emRatingName;
                },
                icons: {
                    name: 'faq',
                    title: "\u6839\u636E\u673A\u6784\u62AB\u9732\u7684\u8BC4\u7EA7\u8BF4\u660E\u5212\u5206\u4E1C\u8D22\u8BC4\u7EA7\u6807\u51C6"
                }
            },
            {
                th: function name(v) {
                    return '报告日期';
                },
                td: function (v) {
                    var publishDate = v.publishDate ? (v.publishDate).substring(5, 11) : '';
                    return publishDate;
                }
            }
        ]
    },
    "zw_stockneweast_stock": {
        config: [
            {
                width: 180,
                th: function name(v) {
                    return '标题';
                },
                td: function (v) {
                    var title = utils_1["default"].txtLeft(v.title, 24, true);
                    return "<a href=\"/report/info/" + v.infoCode + (".html\">" + title + "</a>");
                }
            },
            {
                th: function name(v) {
                    return '东财评级';
                },
                td: function (v) {
                    var emRatingName = v.emRatingName ? v.emRatingName : '';
                    return "" + emRatingName;
                },
                icons: {
                    name: 'faq',
                    title: "\u6839\u636E\u673A\u6784\u62AB\u9732\u7684\u8BC4\u7EA7\u8BF4\u660E\u5212\u5206\u4E1C\u8D22\u8BC4\u7EA7\u6807\u51C6"
                }
            },
            {
                th: function name(v) {
                    return '报告日期';
                },
                td: function (v) {
                    var publishDate = v.publishDate ? (v.publishDate).substring(5, 11) : '';
                    return publishDate;
                }
            }
        ]
    },
    //个股研报正文  ：热门行业评级一览
    "zw_indshotstockpj": {
        theadfix: {
            orgGrade_6: function (v) {
                return '机构投资评级（近六个月）';
            }
        },
        config: [
            {
                width: 50,
                th: function name(v) {
                    return '名称';
                },
                td: function (v) {
                    return "<a href=\"/stockdata/" + v.code + (".html\">" + fieldtools_1["default"].fmtsname(v.name) + "</a>");
                }
            },
            {
                width: 50,
                th: function name(v) {
                    return '涨跌幅';
                },
                td: function (v) {
                    var zdf = v.zdf;
                    var span = '';
                    // span = zdf == '-' ? span = zdf : String(zdf).indexOf('-')>-1 ? span = '<span class="green">' + zdf + '%</span>' : span = '<span class="red">' + zdf + '%</span>';
                    if (zdf > 0) {
                        span = '<span class="red">' + zdf + '%</span>';
                    }
                    else if (zdf < 0) {
                        span = '<span class="green">' + zdf + '%</span>';
                    }
                    else if (zdf == 0) {
                        span = '<span>0.00%</span>';
                    }
                    else if (zdf == '-') {
                        span = '<span>-</span>';
                    }
                    return span;
                    // return '<span class="gghqspan" data-hqtype="zdf" data-market="'+v.market+'" data-code="' + v.stockCode + '"></span>'
                }
            },
            {
                width: 40,
                th: function name(v) {
                    return '研报数';
                },
                td: function (v) {
                    // return !!v.total ? `<a href="">` + v.total+`</a>` :''
                    return v.total !== '' ? v.total : '';
                }
            },
            {
                th: function name(v) {
                    return '买入';
                },
                theadfix: 'orgGrade_6',
                td: function (v) {
                    return v.rateBuy;
                }
            },
            {
                th: function name(v) {
                    return '增持';
                },
                theadfix: 'orgGrade_6',
                td: function (v) {
                    return v.rateIncrease;
                }
            },
            {
                th: function name(v) {
                    return '中性';
                },
                theadfix: 'orgGrade_6',
                td: function (v) {
                    return v.rateNeutral;
                }
            },
            {
                th: function name(v) {
                    return '减持';
                },
                theadfix: 'orgGrade_6',
                td: function (v) {
                    return v.rateReduce;
                }
            },
            {
                th: function name(v) {
                    return '卖出';
                },
                theadfix: 'orgGrade_6',
                td: function (v) {
                    return v.rateSellout;
                }
            }
        ]
    }
};
exports["default"] = config;


/***/ }),

/***/ 3:
/***/ (function(module, exports, __webpack_require__) {

"use strict";

/**
 * 常用公用方法
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
exports.__esModule = true;
var moment_1 = __importDefault(__webpack_require__(0));
var util = {
    //获取几tian之前时间
    timeForMatPreDay: function (num) {
        return moment_1["default"]().add({ "days": 0 - num }).format("YYYY-MM-DD");
    },
    //获取几天之后时间
    timeForMatNextDay: function (num) {
        return moment_1["default"]().add({ "days": num }).format("YYYY-MM-DD");
    },
    //获取几个月之前时间
    timeForMatPreMon: function (num) {
        return moment_1["default"]().add({ "months": num }).format("YYYY-MM-DD");
    },
    //获取几年之前时间
    timeForMatPreYear: function (num) {
        return moment_1["default"]().add({ "years": num }).format("YYYY-MM-DD");
    },
    timeForMatPreStr: function (type, num) {
        var that = this;
        var backtime = "";
        if (type == 'year') {
            backtime = that.timeForMatPreYear(num);
        }
        else if (type == 'month') {
            backtime = that.timeForMatPreMon(num);
        }
        else if (type == 'day') {
            backtime = that.timeForMatPreDay(0 - num);
        }
        return backtime;
    },
    // //获取时间差
    //获取时间差:该方法只能取当前时间及之前时间
    getDateStr: function (preYear, preMonth, preDay) {
        return moment_1["default"]().add({ "years": preYear, "months": preMonth, "days": preDay }).format("YYYY-MM-DD");
    },
    //将标注时间格式转为中文年月日
    getchinesedate: function (str) {
        return moment_1["default"](str).format("YYYY年MM月DD日");
    },
    getUrlParams: function (name) {
        var reg = new RegExp("(^|&)" + name + "=([^&]*)(&|$)", "i");
        var r = window.location.search.substr(1).match(reg);
        // if (r != null) return unescape(r[2]); return null;
        if (r != null)
            return decodeURIComponent(r[2]);
        return null;
    },
    //去除数组重复元素
    uniqueArr: function (arr) {
        var tmp = new Array();
        for (var i in arr) {
            if (tmp.indexOf(arr[i]) == -1) {
                tmp.push(arr[i]);
            }
        }
        return tmp;
    },
    uniqueArr2: function (arr) {
        var ret = [];
        var hash = {};
        for (var i = 0; i < arr.length; i++) {
            var item = arr[i];
            var key = typeof (item) + item;
            if (hash[key] !== 1) {
                ret.push(item);
                hash[key] = 1;
            }
        }
        return ret;
    },
    getstockmarket: function (code) {
        if (code.Length < 3)
            return "sh";
        var one = code.substr(0, 1);
        var three = code.substr(0, 3);
        if (one == "5" || one == "6" || one == "9") {
            return "sh";
        }
        else if (one == "4" || one == "8") {
            return "bj";
        }
        else {
            if (three == "009" || three == "126" || three == "110" || three == "201" || three == "202" || three == "203" || three == "204") {
                return "sh";
            }
            else {
                return "sz";
            }
        }
    },
    formatquteid: function (stockcode, market) {
        if (!stockcode || !market)
            return '';
        var marketcode = '';
        switch (market) {
            case 'SHENZHEN':
                marketcode = 'SZ' + stockcode;
                break;
            case 'SHEN':
                marketcode = 'SZ' + stockcode;
                break;
            case 'SHANGHAI':
                marketcode = 'SH' + stockcode;
                break;
            case 'HU':
                marketcode = 'SH' + stockcode;
                break;
        }
        ;
        return marketcode;
    },
    formatcodeMk: function (stockcode, market) {
        var codeMk = '';
        if (!stockcode || !market)
            return '';
        if (market.indexOf('SHEN') > -1 || market.indexOf('SHENZHEN') > -1) {
            codeMk = stockcode ? stockcode + '2' : '';
        }
        if (market.indexOf('HU') > -1 || market.indexOf('SHANGHAI') > -1) {
            codeMk = stockcode ? stockcode + '1' : '';
        }
        return codeMk;
    },
    formatMarketcode: function (stockcode, market) {
        var codeMk = '';
        if (!stockcode || !market)
            return '';
        if (market.indexOf('SHEN') > -1 || market.indexOf('SHENZHEN') > -1) {
            codeMk = stockcode ? '0.' + stockcode : '';
        }
        if (market.indexOf('HU') > -1 || market.indexOf('SHANGHAI') > -1) {
            codeMk = stockcode ? '1.' + stockcode : '';
        }
        return codeMk;
    },
    //字符串截取
    txtLeft: function (txt, n, needtip) {
        if (txt == null || txt == "") {
            return "-";
        }
        var len = 0;
        for (var i = 0; i < txt.length; i++) {
            if (txt.charCodeAt(i) > 255) {
                len += 2;
            }
            else {
                len++;
            }
            if (len > n + 3) {
                if (needtip) {
                    return '<span title="' + txt + '">' + txt.substring(0, i) + "...</span>";
                }
                else {
                    return txt.substring(0, i) + "..";
                }
                break;
            }
        }
        return txt;
    },
    // 将对象转为params形式的string
    dataToParams: function (data) {
        var arr = [];
        for (var k in data) {
            var value = data[k];
            if (typeof (value) === 'string')
                value = value.replace(/'/g, "%27").replace(/"/g, "%22").replace(/>/g, "%3E").replace(/</g, "%3C").replace(/=/g, "%3D");
            arr.push(k + "=" + value);
        }
        return arr.join("&");
    },
    // 判断字体的颜色，涨：红；跌：绿；无涨跌：黑
    fontColor: function (val) {
        val = ('' + val).replace('%', '');
        if (val > 0) {
            return 'red';
        }
        else if (val < 0) {
            return 'green';
        }
    },
    // 过滤接口返回回来的时间
    changeDate: function (val) {
        var arr = val.split(' ')[0].split('/');
        arr = arr.map(function (item) { return item >= 10 ? item : '0' + item; });
        return arr.join('-');
    },
    // 判断是否是数值，是数值保留两位小数
    toFixed2: function (val, fixed) {
        if (fixed === void 0) { fixed = 2; }
        if (val === '' || val == '-')
            return '-';
        val = parseFloat(val);
        if (isNaN(val))
            return "-";
        return typeof (val) == "number" ? val.toFixed(fixed) : val;
    },
    numFormat: function (value, fixed, unit) {
        if (fixed === void 0) { fixed = 2; }
        if (unit === void 0) { unit = 'auto'; }
        if (value === '' || value == '-')
            return '-';
        value = parseFloat(value);
        if (isNaN(value))
            return "-";
        fixed = typeof fixed == 'undefined' && 2 || fixed;
        var add = '';
        if (unit == "万") {
            value = value / 10000;
        }
        else if (unit == "亿") {
            value = value / 100000000;
        }
        else if (unit == "%") {
            value = value;
            add = '%';
        }
        else if (unit == "auto") {
            if (Math.abs(value) >= 1000000000000) {
                value = value / 1000000000000;
                add = "万亿";
            }
            else if (Math.abs(value) >= 100000000) {
                value = value / 100000000;
                add = "亿";
            }
            else if (Math.abs(value) >= 10000) {
                value = value / 10000;
                add = "万";
            }
        }
        else {
            value = value;
            add = unit;
        }
        value = value.toFixed(fixed);
        return value + add;
    },
    // 取值范围
    getValRegion: function (minVal, maxVal, fixed) {
        if (fixed === void 0) { fixed = 2; }
        var result = "-";
        if (minVal == null) {
            result = this.numFormat(maxVal, fixed);
        }
        else {
            if (maxVal != null) {
                result = minVal == maxVal ? this.numFormat(minVal, fixed) : this.numFormat(minVal, fixed) + "~" + this.numFormat(maxVal, fixed);
            }
            else {
                result = this.numFormat(minVal, fixed);
            }
        }
        return result;
    },
    dateFormat: function (str, type) {
        if (str === '' || str === undefined || str === null || str === '-') {
            return '-';
        }
        try {
            if (/^\d*$/.test(str)) {
                str = parseInt(str);
            }
            type = !!type ? type : 'YYYY-MM-DD';
            var retDate = new Date(str);
            if (isNaN(retDate))
                retDate = this.parseISO8601(str);
            return moment_1["default"](retDate).format(type);
        }
        catch (err) {
            return '-';
        }
    },
    parseISO8601: function (dateStringInRange) {
        var isoExp = /^\s*(\d{4})-(\d\d)-(\d\d)\s*/, date = new Date(NaN), month, parts = isoExp.exec(dateStringInRange);
        if (parts) {
            month = +parts[2];
            date.setFullYear(parts[1], month - 1, parts[3]);
            if (month != date.getMonth() + 1) {
                date.setTime(NaN);
            }
        }
        return date;
    },
    getQueryString: function (name) {
        var reg = new RegExp("(^|&)" + name + "=([^&]*)(&|$)", "i");
        var r = decodeURIComponent(window.location.search).substr(1).match(reg);
        if (r != null)
            return unescape(r[2]);
        return null;
    },
    Substr: function (str, num) {
        var ss;
        // 声明变量。
        if (str.length > num)
            ss = str.substr(0, num) + "...";
        else {
            ss = str;
        }
        // 获取子字符串。
        return ss; // 返回 "Spain"。
    },
    /*
    *判断值是否非空
    */
    getTextValOrEmpty: function (value) {
        if (value != '' && value != undefined && value != null) {
            return value;
        }
        else {
            return '-';
        }
    },
    getDescribe: function (data, num, fix, divide, abs, arr) {
        if (arr === void 0) { arr = null; }
        try {
            if (data === '' || data === undefined || data === null || data === '-' || isNaN(data)) {
                return '<span>-</span>';
            }
            data = parseFloat(data);
            if (data == 0) {
                return "无变化";
            }
            var retult = '';
            if (!arr) {
                arr = ['上升', '下降'];
            }
            if (!divide) {
                retult = this.numFormat(data, num, fix);
            }
            else {
                num = !!parseFloat(num) ? parseFloat(num) : 2;
                if (abs) {
                    retult = Math.abs((data / parseInt(divide))).toFixed(num) + fix;
                }
                else {
                    retult = (data / parseInt(divide)).toFixed(num) + fix;
                }
            }
            var color = '';
            var desc = '';
            color = !!data ? (data == 0 ? '' : data > 0 ? 'red' : 'green') : '';
            desc = !!data ? (data >= 0 ? arr[0] : arr[1]) : '';
            retult = desc + '<span class="' + color + '">' + retult + '</span>';
            return retult;
        }
        catch (err) {
            return '-';
        }
    },
    getColor: function (data) {
        data = parseFloat(data);
        return !!data ? (data == 0 ? '' : data > 0 ? 'red' : 'green') : '';
    },
    getColorByData: function (data, num, fix, divide) {
        try {
            if (data === '' || data === undefined || data === null || data === '-' || isNaN(data)) {
                return '<span>-</span>';
            }
            data = parseFloat(data);
            var retult = '';
            if (!divide) {
                retult = this.FixAmt(data, num, fix);
            }
            else {
                num = !!parseFloat(num) ? parseFloat(num) : 2;
                retult = (data / parseInt(divide)).toFixed(num) + fix;
            }
            var color = '';
            color = !!data ? (data == 0 ? '' : data > 0 ? 'red' : 'green') : '';
            retult = '<span class="' + color + '">' + retult + '</span>';
            return retult;
        }
        catch (err) {
            return '-';
        }
    },
    /*
    *单位自动换算
    */
    FixAmt: function (str, num, fix, ride) {
        try {
            if (str === '' || str === undefined || str === null || str === '-' || isNaN(str)) {
                return '-';
            }
            var result;
            fix = !!fix ? fix : '';
            num = isNaN(parseFloat(num)) ? 2 : parseFloat(num);
            ride = !!ride ? ride : 1;
            str = parseFloat(str) * parseInt(ride);
            var intStr = Math.abs(parseInt(str));
            if (intStr.toString().length > 12) {
                result = (parseFloat(str) / 1000000000000).toFixed(num) + '万亿' + fix;
            }
            else if (intStr.toString().length > 8) {
                result = (parseFloat(str) / 100000000).toFixed(num) + '亿' + fix;
            }
            else if (intStr.toString().length > 4) {
                result = (parseFloat(str) / 10000).toFixed(num) + '万' + fix;
            }
            else {
                result = parseFloat(str).toFixed(num) + fix;
            }
            // console.log(result)
            return result;
        }
        catch (err) {
            return '-';
        }
    },
    /**
     * 静态行情图
     * @param nid
     * @returns
     */
    getWebquotepic: function (nid) {
        var env = this.getUrlParams('myenv') || "";
        if (env == "test") {
            return "https://webquotepictest.eastmoney.com/GetPic.aspx?nid=" + nid + "&imageType=FFRST&type=ffr&token=3a965a43f705cf1d9ad7e1a3e429d622&" + Math.random();
        }
        return "https://webquotepic.eastmoney.com/GetPic.aspx?nid=" + nid + "&imageType=FFRST&type=ffr&token=3a965a43f705cf1d9ad7e1a3e429d622&" + Math.random();
    }
};
exports["default"] = util;


/***/ }),

/***/ 390:
/***/ (function(module, exports, __webpack_require__) {

"use strict";

/**
 * 盈利预测
 *  */
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
exports.__esModule = true;
var datatable_1 = __importDefault(__webpack_require__(5));
var utils_1 = __importDefault(__webpack_require__(3));
var dataurl_1 = __importDefault(__webpack_require__(23));
var dataconfig_1 = __importDefault(__webpack_require__(24));
var datacenter_bks_1 = __webpack_require__(73);
window["tableconfigdatas"] = { tablecolumns: dataconfig_1["default"], tabledataurl: dataurl_1["default"] };
//url参数
var hyId = utils_1["default"].getUrlParams('hyid');
if (hyId)
    hyId = hyId.replace('BK0', '').replace('BK', '');
var marketstr = utils_1["default"].getUrlParams('market');
var markets = ['sha', 'sza', 'kcb', 'cyb', 'bja']; //市场品种目前只有这四种
if (marketstr) {
    marketstr = marketstr.toLowerCase();
}
if (markets.indexOf(marketstr) < 0) {
    marketstr = '';
}
init();
function init() {
    getReportBK(); //行业
}
;
var _datatable = new datatable_1["default"]({
    table_type: 'profitforecast',
    div_id: 'profitforecast_table',
    pager_id: 'profitforecast_table_pager',
    initdata: true,
    initdataone: true,
    enabled_pager: true,
    floathead: true,
    loadingImg: true,
    loadscroll: true,
    pageNoName: "pageNumber",
    initParams: {
        reportName: 'RPT_WEB_RESPREDICT',
        columns: 'WEB_RESPREDICT',
        pageNumber: 1,
        pageSize: 50,
        sortTypes: -1,
        sortColumns: 'RATING_ORG_NUM'
    }
});
//每组保留第一个 firstLetter 字母标志
function filterArr(arr) {
    var temp = [];
    for (var i = 0; i < arr.length; i++) {
        if (temp.indexOf(arr[i].firstLetter) < 0) {
            temp.push(arr[i].firstLetter);
        }
        else {
            arr[i].firstLetter = '';
        }
    }
    return arr;
}
;
function sortLetter(arr, dataLeven) {
    function getValue(option) {
        if (!dataLeven)
            return option;
        var data = option;
        dataLeven.split('.').filter(function (item) {
            data = data[item];
        });
        return data + '';
    }
    arr.sort(function (item1, item2) {
        return getValue(item1).localeCompare(getValue(item2), 'zh-Hans-CN');
    });
    return filterArr(arr);
}
;
function sortBKsLetter(arr, dataLeven) {
    arr.sort(function (item1, item2) {
        return item1[dataLeven].localeCompare(item2[dataLeven], 'zh-Hans-CN');
    });
    var temp = [];
    arr.forEach(function (v) {
        if (temp.indexOf(v[dataLeven]) < 0) {
            temp.push(v[dataLeven]);
        }
        else {
            v[dataLeven] = '';
        }
    });
    return arr;
}
function addFilterDom(backdata) {
    var arr = sortLetter(backdata, 'firstLetter');
    var $li = "";
    var $span = "<li><a class=\"\" data-bkval=\"*\">\u5168\u90E8\u80A1\u7968</a></li>";
    for (var i = 0; i < arr.length; i++) {
        var $b = "";
        var $link = "";
        if (arr[i].firstLetter) {
            $b = "<b>" + arr[i].firstLetter + "</b>";
        }
        $link = "<a class=\"\" target=\"_self\" data-bkval=" + arr[i].bkCode + (">" + arr[i].bkName + "</a>");
        $li += "<li>" + $b + "" + $link + "</li>";
    }
    return $span + $li;
}
;
function addFilterDom2(backdata) {
    var arr = sortBKsLetter(backdata, 'FIRST_LETTER');
    var $li = "";
    var $span = "<li><a class=\"\" data-bkval=\"*\">\u5168\u90E8\u80A1\u7968</a></li>";
    for (var i = 0; i < arr.length; i++) {
        var $b = "";
        var $link = "";
        if (arr[i].FIRST_LETTER) {
            $b = "<b>" + arr[i].FIRST_LETTER + "</b>";
        }
        $link = "<a class=\"\" target=\"_self\" data-bkval=" + arr[i].BOARD_CODE + (">" + arr[i].BOARD_NAME + "</a>");
        $li += "<li>" + $b + "" + $link + "</li>";
    }
    return $span + $li;
}
;
// async function getReportBK() {
//     await bkInfo.hyInfo().then(v => {
//         let backdata = v.data;
//         let $li = addFilterDom(backdata)
//         $('.hy-block').append(`<ul>` + $li + `<ul>`);
//         if (hyId) {
//             $('.hy-block').find('a').each(function () {
//                 let bkval = $(this).data('bkval');
//                 if (bkval == hyId) {
//                     $(this).addClass('select');
//                 }
//             })
//         } else {
//             $('.hy-block').find('li').first().find('a').addClass('select');
//         };
//     });
//     await bkInfo.gnInfo().then(v => {
//         let backdata = v.data;
//         let $li = addFilterDom(backdata)
//         $('.gn-block').append(`<ul>` + $li + `<ul>`);
//     });
//     await bkInfo.dyInfo().then(v => {
//         let backdata = v.data;
//         let $li = addFilterDom(backdata)
//         $('.dy-block').append(`<ul>` + $li + `<ul>`);
//     });
//     setTimeout(temp, 500);
// }
function getReportBK() {
    return __awaiter(this, void 0, void 0, function () {
        var hy_bks, $li, gn_bks, $li3, dy_bks, $li2;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0: return [4 /*yield*/, datacenter_bks_1.getDataCenterBks("2")];
                case 1:
                    hy_bks = _a.sent();
                    $li = addFilterDom2(hy_bks["2"]);
                    $('.hy-block').append("<ul>" + $li + "<ul>");
                    if (hyId) {
                        $('.hy-block').find('a').each(function () {
                            var bkval = $(this).data('bkval');
                            if (bkval == hyId) {
                                $(this).addClass('select');
                            }
                        });
                    }
                    else {
                        $('.hy-block').find('li').first().find('a').addClass('select');
                    }
                    ;
                    return [4 /*yield*/, datacenter_bks_1.getDataCenterBks("3")];
                case 2:
                    gn_bks = _a.sent();
                    $li3 = addFilterDom2(gn_bks);
                    $('.gn-block').append("<ul>" + $li3 + "<ul>");
                    return [4 /*yield*/, datacenter_bks_1.getDataCenterBks("1")];
                case 3:
                    dy_bks = _a.sent();
                    $li2 = addFilterDom2(dy_bks);
                    $('.dy-block').append("<ul>" + $li2 + "<ul>");
                    setTimeout(temp, 500);
                    return [2 /*return*/];
            }
        });
    });
}
// 板块选项卡切换 
$("#title-wrap").on('click', 'li', function () {
    $(this).addClass('active').siblings().removeClass('active');
    $("#bk-content>.child-wrap").eq($(this).index()).addClass("act").siblings().removeClass("act");
});
$("#bk-content > .child-wrap").on('click', 'li', function () {
    $(this).find('a').addClass('select');
    $(this).siblings().find('a').removeClass('select');
    $(this).parents('.child-wrap').siblings().find('li > a').removeClass('select');
    var name = $(this).find('a').text();
    var parentType = $(this).parents('.child-wrap').data('type');
    var filter = '';
    if (name != "全部股票") {
        switch (parentType) {
            case "hy":
                filter = "(INDUSTRY_BOARD=\"" + name + "\")";
                break;
            case "dy":
                filter = "(REGION_BOARD=\"" + name + "\")";
                break;
            case "gn":
                filter = "(CONCEPTINDEX_BOARD like\"%" + name + "%\")";
                break;
            case "market":
                if (name == "沪市A股") {
                    filter = '(MARKET_BOARD in ("069001001001","069001001006"))';
                }
                else if (name == "科创板") {
                    filter = '(MARKET_BOARD="069001001006")';
                }
                else if (name == "深市A股") {
                    filter = '(MARKET_BOARD in ("069001002002","069001002001"))';
                }
                else if (name == "创业板") {
                    filter = '(MARKET_BOARD="069001002002")';
                }
                else if (name == "京市A股") {
                    filter = '(MARKET_BOARD="069001017")';
                }
                break;
            default:
                break;
        }
    }
    $("#profitforecast_table_pager").hide();
    _datatable.showTable({
        search: {
            filter: filter
        },
        pager: 1
    });
});
//控制市场品种参数元素定位
function temp() {
    if (!hyId) {
        //市场品种参数
        if (marketstr && markets.indexOf(marketstr) > -1) {
            $("#title-wrap").find('li').each(function () {
                var type = $(this).data('type');
                if (type == 'market') {
                    $(this).click();
                }
            });
            $("#bk-content > .sc-block").find("li").each(function (e) {
                var val = $(this).find('a').data('bkval');
                if (val == marketstr) {
                    $(this).click();
                    return;
                }
            });
        }
    }
    else {
        var sel = $("a[data-bkval='" + hyId + "']");
        sel.parents(".child-wrap").addClass("act").siblings().removeClass("act");
        sel.parents(".child-wrap").find('a').removeClass('select');
        sel.addClass('select');
        sel.parents(".child-wrap").siblings().find('li > a').removeClass('select');
        var type = sel.parents(".child-wrap").data("type");
        $("#title-wrap li[data-type='" + type + "']").addClass('active').siblings().removeClass('active');
        sel.click();
    }
    ;
}


/***/ }),

/***/ 5:
/***/ (function(module, exports, __webpack_require__) {

"use strict";

/**
 * 数据表格模块
 */
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
exports.__esModule = true;
var webconfig = __webpack_require__(2);
var pager_1 = __importDefault(__webpack_require__(7));
/**
 * 三种排序类型的切换
 * @param thistype
 */
function getSortType(thistype) {
    var backtype = '';
    switch (thistype) {
        case 'default':
            backtype = 'desc';
            break;
        case 'desc':
            backtype = 'asc';
            break;
        case 'asc':
            backtype = 'default';
            break;
    }
    return backtype;
}
/**
 * 判断是否需要第一层thead
 * @param config
 */
function isFixThead(config) {
    var back = false;
    config.config.forEach(function (v) {
        if (v.theadfix != undefined) {
            back = true;
        }
    });
    return back;
}
/**
 * 计算接下来有几个相同的theadfix
 * @param config
 * @param configindex
 */
function getSameTheadFix(config, configindex) {
    if (configindex >= config.length - 1) {
        return 1;
    }
    var count = 1;
    var theadfix_val = config[configindex].theadfix;
    for (var index = configindex + 1; index < config.length; index++) {
        if (theadfix_val == config[index].theadfix) {
            count++;
        }
        else {
            break;
        }
    }
    return count;
}
var datatable = /** @class */ (function () {
    function datatable(options) {
        this.table_type = '';
        this.div_id = '';
        this.thissort = ''; //排序字段
        this.this_sort_type = 'default'; //排序类型
        this.thisindex = 1; //当前页数
        this.enabled_pager = true; //是否翻页
        this.pager_id = '';
        this.tablepager = null; //分页示例
        this.floathead = false; //是否开启头部固顶
        this.tableclass = 'table-model';
        this.curSortItem = '';
        this.curSortParam = '';
        this.onShowComplete = null;
        this.initsearch = false;
        this.loadscroll = false;
        this.loadingImg = false;
        this.initdata = false;
        this.initdataLab = 'initdata'; //该字段为首页缓存在页面赋值的字段,默认'initdata'
        this.initdataone = false;
        this.tablecolumns = {};
        this.tabledataurl = {};
        this.postUrl = false;
        this.webhostname = '';
        this.table_type = options.table_type;
        this.div_id = options.div_id;
        this.enabled_pager = options.enabled_pager != undefined ? options.enabled_pager : true; //是否开启翻页功能
        this.pageNoName = options.pageNoName;
        this.pager_id = options.pager_id;
        this.floathead = options.floathead != undefined ? options.floathead : false; //默认false
        this.tableclass = options.tableclass != undefined ? options.tableclass : 'table-model';
        this.onShowComplete = options.onShowComplete || null;
        this.loadscroll = options.loadscroll || false;
        this.loadingImg = options.loadingImg || false;
        this.initdata = options.initdata || false;
        this.initdataLab = options.initdataLab || 'initdata';
        this.initdataone = options.initdataone || false;
        // 是否是post请求
        this.postUrl = options.postUrl || false;
        var initParams = options.initParams || false;
        // 获取table的config和theadfix，入口已配置，用入口的，入口没有配置，用dataconfig文件里面的
        this.tablecolumns = options.columns ? options.columns : (tableconfigdatas ? tableconfigdatas.tablecolumns[this.table_type] : {});
        // 获取table的dataurl，入口已配置，用入口的，入口没有配置，用dataurl文件里面的
        this.tabledataurl = options.dataurl ? options.dataurl : (tableconfigdatas ? tableconfigdatas.tabledataurl[this.table_type] : {});
        // 获取数据的hostname，没有取默认的‘dataurl’，dataurl在webconfig里面表示的是研报api的host
        this.webhostname = this.tabledataurl.hostname ? this.tabledataurl.hostname : (options.dataurl ? (options.dataurl.hostname ? options.dataurl.hostname : 'dataurl') : 'dataurl');
        if (initParams) {
            this.search(initParams);
        }
        else {
            this.init();
        }
        if (this.floathead) {
            var _this_1 = this;
            setTimeout(function () {
                _this_1.makeFloatHead();
            }, 1000);
        }
    }
    datatable.prototype.init = function () {
        return __awaiter(this, void 0, void 0, function () {
            var _this;
            return __generator(this, function (_a) {
                _this = this;
                _this.showTable({});
                return [2 /*return*/];
            });
        });
    };
    datatable.prototype.showTable = function (options) {
        return __awaiter(this, void 0, void 0, function () {
            var _this, loadhtml, data;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        _this = this;
                        $('#' + _this.div_id).css({ 'position': 'relative' });
                        if (_this.loadingImg) {
                            loadhtml = '<div class="table-loading"><img src="/newstatic/images/loading.gif"></img></div>';
                            $('#' + _this.div_id).append(loadhtml);
                        }
                        ;
                        return [4 /*yield*/, this.getData(options)["catch"](function () {
                                return null;
                            })];
                    case 1:
                        data = _a.sent();
                        $('#' + this.div_id).empty().append(this.makeTable(data));
                        if (this.onShowComplete) {
                            this.onShowComplete(data);
                        }
                        ;
                        $('.table-loading').css({ 'display': 'none !importannt' });
                        return [2 /*return*/];
                }
            });
        });
    };
    datatable.prototype.getData = function (options) {
        var _a, _b, _c, _d;
        var _this = this;
        var params = this.tabledataurl['params'];
        if (options.sort != undefined) {
            var sortobj = options.sort;
            var nextsort = getSortType(this.this_sort_type);
            this.this_sort_type = nextsort;
            if (sortobj[nextsort])
                params = $.extend(params, (_a = {},
                    _a[sortobj[nextsort].sortkey] = sortobj[nextsort].value,
                    _a));
            if (sortobj.sortname) {
                params = $.extend(params, (_b = {},
                    _b[sortobj.sortname.key] = sortobj.sortname.value,
                    _b));
            }
            if (sortobj.sortname) {
                params = $.extend(params, (_c = {},
                    _c[sortobj.sortname.key] = sortobj.sortname.value,
                    _c));
            }
        }
        if (options.search) {
            params = $.extend(params, options.search);
        }
        if (options.pager) {
            params = $.extend(params, {
                p: options.pager,
                pageNo: options.pager,
                pageNum: options.pager,
                pageNumber: options.pager //比如：新版datacenter
            });
        }
        // 特殊分页字段，支持动态传分页字段名
        // 比如：高管持股页面分页（url: /executive/gdzjc-jzc.html）
        if (_this.pageNoName) {
            params = $.extend(params, (_d = {},
                _d[_this.pageNoName] = options.pager,
                _d));
        }
        return this.getDataAjax(params);
    };
    datatable.prototype.getDataAjax = function (params) {
        return __awaiter(this, void 0, void 0, function () {
            //ajax请求抽出\
            function getDataInfos() {
                if (_this.postUrl) {
                    try {
                        var ajaxfn = {
                            url: url,
                            type: 'POST',
                            timeout: 20000,
                            jsonpCallback: rnd,
                            data: JSON.stringify(params),
                            dataType: "json",
                            contentType: "application/json",
                            error: function (msg) {
                                $('.stockdata #' + _this.div_id).parents('.table-cont').hide();
                            }
                        };
                        // 地址为三期二期的接口，需要传jsonp参数
                        _this.webhostname == 'datainterface' ? (ajaxfn['jsonp'] = 'cb') : undefined;
                        return $.ajax(ajaxfn);
                    }
                    catch (error) {
                        return [];
                    }
                }
                else {
                    try {
                        var dataType = _this.webhostname == 'url' ? 'json' : 'jsonp';
                        var ajaxfn = {
                            url: url,
                            type: 'GET',
                            timeout: 20000,
                            jsonpCallback: rnd,
                            dataType: dataType,
                            data: params,
                            error: function (msg) {
                                // console.log(_this.div_id)
                                $('.stockdata #' + _this.div_id).parents('.table-cont').hide();
                            }
                        };
                        // 地址为三期二期的接口，需要传jsonp参数
                        _this.webhostname == 'datainterface' ? (ajaxfn['jsonp'] = 'cb') : undefined;
                        return $.ajax(ajaxfn);
                    }
                    catch (error) {
                        return [];
                    }
                }
            }
            function hasData(data) {
                if (data == null) {
                    $('#' + _this.div_id).parents('.table-cont').hide();
                }
                else if (data.length <= 0) {
                    // 隐藏对应的table
                    $('#' + _this.div_id).parents('.table-cont').hide();
                }
            }
            var _this, rnd, url, getdatafunction, onecache, dtd;
            var _this_1 = this;
            return __generator(this, function (_a) {
                _this = this;
                rnd = 'datatable' + Math.floor(Math.random() * 10000000 + 1);
                url = webconfig.getWebPath(this.webhostname) + this.tabledataurl['url'];
                getdatafunction = null;
                if (this.initdata) {
                    this.initdata = false;
                    onecache = (function () {
                        if (window.location.search.length > 1) { //判断url?后面是否带有参数
                            return true;
                        }
                        else {
                            return false;
                        }
                    })();
                    if (this.initdataone && onecache) { //如果initdataone&&onecache 则走缓存逻辑:为了处理有些页面带参数和不带参数(首次请求走接口)
                        getdatafunction = getDataInfos();
                    }
                    else {
                        dtd = $.Deferred();
                        getdatafunction = dtd.resolve(window[this.initdataLab]); //抓取页面缓存的数据initdata
                    }
                }
                else { //如不需要缓存逻辑 直接走接口
                    getdatafunction = getDataInfos();
                }
                try {
                    return [2 /*return*/, getdatafunction.then(function (v) {
                            // 如果api为datacenter的接口数据，返回的数据直接掉result层
                            if (_this_1.webhostname == 'datacenter' || _this_1.webhostname == 'datacentertest') {
                                v = v.result;
                            }
                            // 如果api为data的接口数据，返回的数据用re
                            if (_this_1.webhostname == 'url') {
                                v.data = v.re ? v.re : v.data;
                            }
                            if (_this_1.enabled_pager && v) {
                                if (v.pages > 1) {
                                    // 如果是股市日历相关分页
                                    _this.pager(v.pages * 50, params[_this.pageNoName || 'pageNo']);
                                }
                                else {
                                    // 如果是普通分页
                                    _this.pager(v.hits, params[_this.pageNoName || 'pageNo']);
                                }
                                //@ts-ignore
                                if (v.data && v.data != [] && (v.TotalPage > 1 || v.pages > 1) && v.data.length > 0) {
                                    $('#' + _this_1.pager_id).show();
                                }
                                else {
                                    $('#' + _this_1.pager_id).hide();
                                }
                            }
                            // 如果是api为dcfmurl的接口数据，返回的数据要处理一下
                            if (_this_1.webhostname == 'dcfmurl' && !v.font) {
                                hasData(v);
                                if (v.data)
                                    return v;
                                return { data: v };
                            }
                            // api为dcfmurl，且部分字段需要转码的情况
                            if (_this_1.webhostname == 'dcfmurl' && v.font) {
                                hasData(v.data);
                                var str_1 = JSON.stringify(v.data);
                                var font = v.font.FontMapping;
                                if (font) {
                                    font.map(function (item) {
                                        str_1 = str_1.replace(new RegExp(item.code, "g"), item.value);
                                    });
                                }
                                return { data: JSON.parse(str_1) };
                            }
                            // 如果是api为datainterface的接口数据，数据返回为空的时候需要特殊处理
                            if (_this_1.webhostname == 'datainterface') {
                                if (v.data[0].stats == false) {
                                    return [];
                                }
                            }
                            v ? hasData(v.data) : hasData(v);
                            return v;
                        })];
                }
                catch (error) {
                    return [2 /*return*/, null];
                }
                return [2 /*return*/];
            });
        });
    };
    ;
    //生成表格头部
    datatable.prototype.makeTableHead = function (data) {
        var _this_1 = this;
        var _this = this;
        //表格头部
        var table_html_thead = $("<thead></thead>");
        var thead_tr1 = $('<tr></tr>');
        var thead_tr2 = $('<tr></tr>');
        var need_theadfix = isFixThead(this.tablecolumns); //判断是否需要第一层thead
        var ctheadfix = ''; //当前的theadfix
        this.tablecolumns['config'].forEach(function (configitem, index) {
            var thead_th = $('<th></th>');
            if (configitem.width != undefined) {
                thead_th.attr('width', configitem.width + 'px');
            }
            if (configitem.paramname != undefined) { //给当前排序标签添加name
                thead_th.attr('data-name', configitem.paramname);
            }
            //排序
            if (configitem.sort != undefined) {
                var sorthtml_1 = $('<a target="_self" href="javascript:;">' + configitem.th({ data: data }) + '</a>');
                //给头部th 加icon
                if (configitem.icons != undefined) {
                    sorthtml_1.append('<span class="icon icon_' + configitem.icons.name + ' vbottom" title="' + configitem.icons.title + '"></span>');
                }
                var defaultsortType = configitem.defaultsortType; //默认排序类型
                if (defaultsortType != undefined && !_this.curSortParam) { //默认排序只执行第一次
                    if (defaultsortType == 'asc') {
                        sorthtml_1.append('<span class="icon icon_asc sorticon"></span>');
                    }
                    else if (defaultsortType == 'desc') {
                        sorthtml_1.append('<span class="icon icon_desc sorticon"></span>');
                    }
                }
                ;
                var sortnamename = configitem.paramname;
                if (sortnamename && _this.curSortParam == sortnamename) { //判断是否是当前排序字段
                    if (_this_1.this_sort_type == 'asc') {
                        sorthtml_1.append('<span class="icon icon_asc sorticon"></span>');
                    }
                    else if (_this_1.this_sort_type == 'desc') {
                        sorthtml_1.append('<span class="icon icon_desc sorticon"></span>');
                    }
                    //控制浮动表头的样式切换 
                    if (_this.floathead) {
                        var floatTableTh = $('#' + _this.div_id + 'float').find('th');
                        floatTableTh.each(function (index, el) {
                            if ($(this).data('name') == _this.curSortParam) {
                                $(this).find('a').empty().append(sorthtml_1.clone(true));
                            }
                            else {
                                $(this).find('.sorticon').remove();
                            }
                        });
                    }
                    ;
                }
                ;
                thead_th.append(sorthtml_1);
                sorthtml_1.on('click', function () {
                    if (_this.curSortParam != $(this).parent().data('name')) {
                        _this.this_sort_type = 'default';
                    }
                    ; //如果不是当前排序字段，则将排序类型初始化
                    _this.curSortParam = $(this).parent().data('name'); //存储全局 记录当前排序字段
                    // if (defaultsortType && defaultsortType != 'default'){  //处理带默认排序的排序类型
                    //   _this.this_sort_type = defaultsortType
                    //   configitem.defaultsortType = false
                    //   defaultsortType = false
                    // }
                    _this.showTable({
                        sort: configitem.sort,
                        pager: 1
                    });
                });
            }
            else {
                thead_th.append(configitem.th({
                    data: data
                })); //整个传入，具体地方具体处理
                //给头部th 加icon
                if (configitem.icons != undefined) {
                    thead_th.append('<span class="icon icon_' + configitem.icons.name + ' vbottom" title="' + configitem.icons.title + '"></span>');
                }
            }
            //是否多级头部处理 
            if (need_theadfix) {
                if (configitem.theadfix != undefined) {
                    //thead_tr2.append(thead_th)
                    if (configitem.theadfix != ctheadfix) {
                        var fixcount = getSameTheadFix(_this_1.tablecolumns['config'], index);
                        var theadfixth = $('<th colspan="' + fixcount + '">' + _this_1.tablecolumns['theadfix'][configitem.theadfix](data) + '</th>');
                        //thead_th.attr('colspan', fixcount)
                        ctheadfix = configitem.theadfix;
                        thead_tr1.append(theadfixth);
                    }
                    thead_tr2.append(thead_th);
                    //thead_tr1.append(thead_th)
                }
                else {
                    thead_th.attr('rowspan', 2);
                    thead_tr1.append(thead_th);
                }
            }
            else {
                thead_tr1.append(thead_th);
            }
        });
        if (need_theadfix) {
            table_html_thead.append(thead_tr1).append(thead_tr2);
        }
        else {
            table_html_thead.append(thead_tr1);
        }
        return table_html_thead;
    };
    ;
    datatable.prototype.makeTable = function (data) {
        var _this = this;
        var table_html = $('<table class=' + this.tableclass + '><thead></thead><tbody></tbody></table>');
        var table_html_thead = _this.makeTableHead(data);
        $('thead', table_html).replaceWith(table_html_thead);
        if (data && data.data != undefined && data.data.length > 0) {
            data.data.forEach(function (v, index) {
                //特殊逗号分开数据的接口先转换为对象，省去每次split
                if (typeof (v) == 'string') {
                    var splitstr = ',';
                    var arr = v.split(splitstr);
                    var model_1 = {};
                    arr.forEach(function (s, index) {
                        model_1['f' + index] = s;
                    });
                    v = model_1;
                }
                // 数据为对象才添加v.index，不然有的string数据报错
                typeof (v) == 'object' ? (v.index = index + Number(data.pageNo ? (data.pageNo - 1) * 50 : '')) : '';
                var tr = $('<tr></tr>');
                //表格内容
                _this.tablecolumns['config'].forEach(function (configitem) {
                    var _a, _b;
                    var isdefaultsort = configitem.defaultsortType && configitem.defaultsortType != 'default';
                    var iscursort = _this.curSortParam == configitem.paramname && _this.this_sort_type != 'default';
                    var td = {};
                    if (_this.curSortParam) {
                        td = iscursort ? $('<td class="col"></td>') : $('<td></td>');
                    }
                    else {
                        td = isdefaultsort ? $('<td class="col"></td>') : $('<td></td>');
                    }
                    td.html(configitem.td(v, (_b = (_a = _this === null || _this === void 0 ? void 0 : _this.tablepager) === null || _a === void 0 ? void 0 : _a.ops) === null || _b === void 0 ? void 0 : _b.currentpage));
                    tr.append(td);
                });
                $('tbody', table_html).append(tr);
            });
        }
        else {
            var len = this.tablecolumns['config'].length;
            var tr = $('<tr><td colspan=' + len + ' class="none">暂无数据</td></tr>');
            $('tbody', table_html).append(tr);
        }
        return table_html;
    };
    datatable.prototype.pager = function (sum, currentPage) {
        var _this = this;
        if (_this.tablepager == null) {
            _this.tablepager = new pager_1["default"]({
                pagerid: this.pager_id,
                currentpage: currentPage ? currentPage : 1,
                pagesize: 50,
                allcount: sum,
                href: '',
                onClick: function (index) {
                    _this.goPager(index);
                }
            });
        }
        else {
            _this.tablepager.ops.allcount = sum;
            _this.tablepager.ops.currentpage = currentPage ? currentPage : 1;
            _this.tablepager.show();
        }
        // _this.tablepager.show()
    };
    datatable.prototype.goPager = function (toindex) {
        return __awaiter(this, void 0, void 0, function () {
            var _top;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        //是否开启加载滚动
                        if (this.loadscroll && $('.framecontent').length) {
                            _top = $('.framecontent').offset().top;
                            $("html,body").animate({ "scrollTop": _top });
                        }
                        return [4 /*yield*/, this.showTable({
                                pager: toindex
                            })];
                    case 1:
                        _a.sent();
                        return [2 /*return*/];
                }
            });
        });
    };
    datatable.prototype.search = function (searchobj) {
        return __awaiter(this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.showTable({
                            search: searchobj
                        })];
                    case 1:
                        _a.sent();
                        return [2 /*return*/];
                }
            });
        });
    };
    //生成浮动表头
    datatable.prototype.makeFloatHead = function () {
        var _this = this;
        $('#' + this.div_id + 'float').empty().remove();
        var floatTable = $('<div id="' + this.div_id + 'float" class="table-float-head"><table class="floathead table-model" style="margin-top:0;width:865px"><thead></thead></table></div>');
        var _thead = $('#' + this.div_id + ' table thead');
        $('thead', floatTable).append($('#' + this.div_id + ' table thead').clone(true));
        $('tr', _thead).find('th').each(function (index, el) {
            $('tr', floatTable).find('th').eq(index).width($(el).width());
        });
        $('#' + this.div_id).before(floatTable);
        $(window).scroll(function () {
            var _tableheight = $('#' + _this.div_id).height();
            var _offsettop = $('#' + _this.div_id).offset().top;
            var _scrollheight = $(window).scrollTop();
            var floathtml = $('#' + _this.div_id + 'float');
            if (_scrollheight > _offsettop) {
                floathtml.css({ 'display': 'block' });
                if (_scrollheight > (_offsettop + _tableheight) - 50) {
                    floathtml.css({ 'display': 'none' });
                }
            }
            else {
                floathtml.css({ 'display': 'none' });
            }
            // X轴滚动时动态设置left值
            var _scrollleft = $(window).scrollLeft();
            floathtml.css({ 'left': (_scrollleft ? 135 - _scrollleft : 'auto') });
        });
    };
    ;
    return datatable;
}());
exports["default"] = datatable;


/***/ }),

/***/ 7:
/***/ (function(module, exports, __webpack_require__) {

"use strict";

/**
 * 分页组件
 */
exports.__esModule = true;
var pagerbox = /** @class */ (function () {
    function class_1(opts) {
        var _this = this;
        this.pageshownum = 5;
        this.target = "";
        var defaults = {
            pagerid: '',
            currentpage: 1,
            pagesize: 0,
            allcount: 0,
            href: '',
            onClick: null
        };
        this.ops = $.extend(defaults, opts);
        if (!this.ops.pagerid) {
            if (!this.ops.dom)
                return;
            this.div = this.ops.dom;
        }
        else
            this.div = $('#' + this.ops.pagerid);
        var formhtml = $("<div class=\"gotopage\"><form target=\"_self\"><span class='tab'>\u8DF3\u8F6C\u5230</span><input class='ipt' type='text' id=\"gotopageindex\"><input type='submit' class='btn' value='\u786E\u5B9A'></form></div>");
        formhtml.on('submit', function name() {
            var gotopageindex = 1;
            var totalpage = _this.ops.totalPage || Math.ceil(_this.ops.allcount / _this.ops.pagesize);
            try {
                gotopageindex = parseInt($.trim($('#gotopageindex', formhtml).val().toString()));
            }
            catch (error) {
            }
            //如果超过最大页数
            if (totalpage < gotopageindex) {
                gotopageindex = totalpage;
            }
            if (gotopageindex < 1) {
                gotopageindex = 1;
            }
            _this.changePage(gotopageindex);
            return false;
        });
        this.div.html('').append('<div class="pagerbox"></div>').append(formhtml);
        this.bind();
        this.show();
    }
    ;
    class_1.prototype.changePage = function (curPage) {
        var _this = this;
        if (!!_this.ops.href) {
            // console.log(this.ops.href)       
            return false;
        }
        else if (this.ops.onClick) {
            _this.ops.onClick(curPage);
        }
    };
    ;
    class_1.prototype.show = function () {
        var _this = this;
        var pageshownum = _this.pageshownum;
        var totalPage = this.ops.totalPage || Math.ceil(this.ops.allcount / this.ops.pagesize);
        var currentpage = _this.ops.currentpage;
        var hrefStr = _this.ops.href ? _this.ops.href : 'javascript: ;';
        var target = _this.target ? _this.target : '_self';
        var leftStr = "";
        var rightStr = "";
        var allStr = "";
        if (pageshownum <= 0)
            pageshownum = 5;
        var LRNum = Math.ceil((pageshownum - 1) / 2);
        // var totalPage = this.totalPage;
        var Li = 0, Rj = 0;
        //需要展示的页数和当前页逻辑处理
        for (var i = LRNum; i >= 1; i--) {
            if (currentpage - i >= 1) {
                leftStr += "<a target=" + target + " href=" + hrefStr + " data-page=" + (currentpage - i) + ">" + (currentpage - i) + "</a>";
                Li += 1;
            }
        }
        for (var j = 1; j <= LRNum; j++) {
            if (currentpage + j <= totalPage) {
                rightStr += "<a target=" + target + " href=" + hrefStr + " data-page=" + (currentpage + j) + ">" + (currentpage + j) + "</a>";
                Rj += 1;
            }
        }
        //总页数小于/大于显示页数时处理
        if (totalPage < pageshownum)
            pageshownum = totalPage;
        if ((Li + Rj) < (pageshownum - 1)) {
            if (currentpage <= pageshownum / 2 + 1) {
                for (var j = currentpage + Rj + 1; j <= pageshownum; j++) {
                    rightStr += "<a target=" + target + " href=" + hrefStr + " data-page=" + j + ">" + j + "</a>";
                }
            }
            else {
                for (var i = (currentpage - Li - 1); i > totalPage - pageshownum; i--) {
                    leftStr = "<a target=" + target + " href=" + hrefStr + " data-page=" + i + ">" + i + "</a>" + leftStr;
                }
            }
        }
        if (totalPage > pageshownum) {
            if (currentpage > (pageshownum + 1) / 2) {
                if (currentpage <= pageshownum) {
                    leftStr = "<a target=" + target + " href=" + hrefStr + " data-page='1'>1</a>" + "<a href=" + hrefStr + " target=" + target + "  data-page=" + 1 + ">...</a>" + leftStr;
                }
                else {
                    leftStr = "<a target=" + target + " href=" + hrefStr + " data-page='1'>1</a>" + "<a href=" + hrefStr + " target=" + target + "  data-page=" + (currentpage - 5) + ">...</a>" + leftStr;
                }
                // leftStr = "<a target=" + target + " href=" + hrefStr + " data-page='1'>1</a>" + "<a href=" + hrefStr + " target=" + target + "  data-page=" + (currentpage -5) +">...</a>" + leftStr;
            }
            if (currentpage <= totalPage - (pageshownum + 1) / 2) {
                if ((totalPage - currentpage) < pageshownum) {
                    rightStr += "<a target=" + target + " href=" + hrefStr + " data-page=" + (totalPage) + ">...</a>" + "<a href=" + hrefStr + " target=" + target + " data-page=" + totalPage + ">" + totalPage + "</a>";
                }
                else {
                    rightStr += "<a target=" + target + " href=" + hrefStr + " data-page=" + (currentpage + 5) + ">...</a>" + "<a href=" + hrefStr + " target=" + target + " data-page=" + totalPage + ">" + totalPage + "</a>";
                }
                //rightStr += "<a target=" + target + " href=" + hrefStr + " data-page=" + (currentpage+5)+">...</a>" + "<a href=" + hrefStr + " target=" + target + " data-page=" + totalPage+">" + totalPage + "</a>";
            }
        }
        //上一页
        var prevPage = "";
        var pre = currentpage - 1;
        if (pre < 1) {
            pre = 1;
        }
        prevPage = _this.ops.href ? _this.ops.href : 'javascript: ;';
        if (pre == 1) {
            //prevPage = '';
        }
        else {
            // prevPage = "_" + pre;
            // if (!!this.ops.href) {
            //     prevPage = "" + pre;
            // }
        }
        //下一页
        var nextPage = "";
        var next = currentpage + 1;
        if (next > totalPage) {
            next = totalPage;
        }
        nextPage = _this.ops.href ? _this.ops.href : 'javascript: ;';
        if (next == 1) {
            //nextPage = '';
        }
        else {
            // nextPage = "_" + next;
            // if (!!this.ops.href) { 
            //     nextPage = "" + next;
            // }
        }
        //上一页下一页及指定页数跳转
        if (currentpage > 1) {
            allStr = "<a target=" + target + " href=" + prevPage + " data-page=" + pre + ">上一页</a> ";
        }
        allStr += leftStr + "<a class='active' data-page=" + currentpage + ">" + currentpage + "</a>" + rightStr;
        if (currentpage < totalPage) {
            allStr += " <a target=" + target + " href=" + nextPage + " data-page=" + next + ">下一页</a>";
        }
        $('.pagerbox', this.div).empty().append(allStr);
        $('#gotopageindex', this.div).val(currentpage);
    };
    class_1.prototype.bind = function () {
        var _this = this;
        this.div.off('click').on('click', 'a', function () {
            var curPage = $(this).data('page');
            _this.changePage(curPage);
        });
    };
    return class_1;
}());
exports["default"] = pagerbox;


/***/ }),

/***/ 73:
/***/ (function(module, exports, __webpack_require__) {

"use strict";

var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
exports.__esModule = true;
exports.getDataCenterBks = void 0;
var webconfig = __webpack_require__(2);
function formatBKLevel(bks) {
    try {
        var back = bks.reduce(function (pre, cur) {
            if (pre[cur.BOARD_LEVEL]) {
                pre[cur.BOARD_LEVEL].push(cur);
            }
            else {
                pre[cur.BOARD_LEVEL] = [cur];
            }
            return pre;
        }, {});
        return back;
    }
    catch (error) {
        return bks;
    }
}
/**
 * 数据中心板块获取 web
 * @param type 1地区2行业3概念
 */
function getDataCenterBks(type) {
    var _a, _b;
    return __awaiter(this, void 0, void 0, function () {
        var res, error_1;
        return __generator(this, function (_c) {
            switch (_c.label) {
                case 0:
                    _c.trys.push([0, 2, , 3]);
                    return [4 /*yield*/, $.ajax({
                            url: webconfig.getWebPath('datacenter') + 'api/data/v1/get',
                            type: 'GET',
                            dataType: 'jsonp',
                            // jsonp: "cb",
                            data: {
                                reportName: "RPT_EMBOARD_ALL",
                                columns: "BOARD_CODE,BOARD_NAME,BOARD_CODE_BK,BOARD_LEVEL,BOARD_TYPE,BOARD_TYPE_NAME,FIRST_LETTER",
                                quoteColumns: "",
                                sortColumns: "FIRST_LETTER",
                                sortTypes: 1,
                                source: "WEB",
                                client: "WEB",
                                filter: "(BOARD_TYPE=" + type + ")",
                                pageNumber: 1,
                                pageSize: 1000,
                            }
                        })];
                case 1:
                    res = _c.sent();
                    if (((_b = (_a = res === null || res === void 0 ? void 0 : res.result) === null || _a === void 0 ? void 0 : _a.data) === null || _b === void 0 ? void 0 : _b.length) > 0) {
                        // 只有行业板块有分级
                        if (type == "2") {
                            return [2 /*return*/, formatBKLevel(res.result.data)];
                        }
                        return [2 /*return*/, res.result.data];
                    }
                    return [2 /*return*/, null];
                case 2:
                    error_1 = _c.sent();
                    return [2 /*return*/, null];
                case 3: return [2 /*return*/];
            }
        });
    });
}
exports.getDataCenterBks = getDataCenterBks;


/***/ })

/******/ });
//# sourceMappingURL=profitforecast.js.map