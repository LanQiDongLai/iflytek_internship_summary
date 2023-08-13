#include <crow.h>

#include <sstream>

#include <sys/socket.h>

#include <curl/curl.h>

extern "C" {
#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"
}


struct ExampleMiddleware {
    std::string message;

    ExampleMiddleware() {
        message = "";
    }

    void setMessage(std::string newMsg) {
        message = newMsg;
    }

    struct context {
    };

    void before_handle(crow::request& req, crow::response& res, context& ctx) {
        CROW_LOG_DEBUG << " - MESSAGE: " << message;
    }

    void after_handle(crow::request& req, crow::response& res, context& ctx) {

    }
};


size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* response) {
    size_t total_size = size * nmemb;
    response->append(static_cast<char*>(contents), total_size);
    return total_size;
}

int RelayToServe(lua_State *L) {
    int port = lua_tointeger(L, -1);
    std::string ip = lua_tostring(L, -2);
    int body_len = lua_tointeger(L, -3);
    const char* body = lua_tostring(L, -4);
    std::string path = lua_tostring(L, -5);
    std::string method = lua_tostring(L, -6);
    std::string response;
    int response_code;

    // 初始化libcurl
    curl_global_init(CURL_GLOBAL_ALL);
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Failed to initialize libcurl." << std::endl;
        exit(EXIT_FAILURE);
    }

    // 设置URL
    std::string url = "http://" + ip + ":" + std::to_string(port) + path;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    
    // 设置请求字段
    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/octet-stream");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    // 设置请求方法
    curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, method.c_str());

    // 设置请求body
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, body_len);

    // 设置接收响应数据的回调函数
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    // 发送请求
    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        std::cerr << "Failed to send HTTP request: " << curl_easy_strerror(res) << std::endl;
    }

    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);

    // 清理并释放资源
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    lua_pushstring(L, response.c_str());
    lua_pushinteger(L, response_code);
    return 2;
}

int main() {
    crow::App<ExampleMiddleware> app;

    app.get_middleware<ExampleMiddleware>().setMessage("hello");


    app.route_dynamic("/<path>").methods(crow::HTTPMethod::GET, crow::HTTPMethod::POST, crow::HTTPMethod::PUT, crow::HTTPMethod::DELETE)
        ([](const crow::request& req, crow::response& resp, std::string path) {
        CROW_LOG_DEBUG << "Mark Body:" << req.body;
        lua_State* L = luaL_newstate();
        luaL_openlibs(L);

        lua_register(L, "RelayToServe", RelayToServe);

        if (luaL_dofile(L, "Relay.lua") != LUA_OK) {
            std::cout << "lua文件读取失败" << std::endl;
            exit(EXIT_FAILURE);
        }
        lua_getglobal(L, "Relay");
        const char* method;
        if (req.method == crow::HTTPMethod::GET)
            method = "GET";
        if (req.method == crow::HTTPMethod::POST)
            method = "POST";
        if (req.method == crow::HTTPMethod::PUT)
            method = "PUT";
        if (req.method == crow::HTTPMethod::DELETE)
            method = "DELETE";
        CROW_LOG_DEBUG << "Mark Path:" << req.raw_url;
        lua_pushstring(L, method);
        lua_pushstring(L, req.raw_url.c_str());
        lua_pushlstring(L, req.body.c_str(), req.body.length());
        int res = LUA_OK;
        res = lua_pcall(L, 3, 2, 0);
        CROW_LOG_DEBUG << "Mark Res:" << res;
        if (res != LUA_OK) {
            std::cout << "Error" << std::endl;
            std::cout << lua_tostring(L, -1) << std::endl;
            lua_close(L);
        }
        int code_reponse = lua_tointeger(L, -1);
        std::string json_response = lua_tostring(L, -2);
        CROW_LOG_DEBUG << "Mark Json Resp:" << json_response;
        resp.write(json_response);
        resp.end();

        lua_close(L);
            }
    );

    crow::logger::setLogLevel(crow::LogLevel::DEBUG);

    app.port(18080)
        .multithreaded()
        .run();
    return 0;
}
