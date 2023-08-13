package.cpath = package.cpath .. ";/home/LanQiDongLai/Software/LuaRocks/lib/lua/5.4/?.so"

local lrex = require("rex_posix")
local socket = require("socket")

---解析路径对应的服务器列表
---@param path string
---@return integer server_count
---@return table server_list
PathGetServe = function (path)
    local matches = {lrex.match(path, "/([^#?&/]*)")}
    if #matches <= 0 then
        error("解析路径对应服务器失败")
    end
    return 1, matches
end

---将路径和载荷按照对应的服务器进行转换
---@param path string
---@param body string
---@param serve string
---@return string converted_path
---@return string converted_body
PathBodyConvert = function (path, body, serve)
    if serve == "pytorch" then
        local matches = {lrex.match(path, "/([^#?&/]*)/([^#?&/]*)(/[^ ]*)")}
        if #matches <= 0 then
            error("解析路径失败")
        end
        return matches[3], body
    elseif serve == "tensorflow" then
        local matches = {lrex.match(path, "/([^#?&/]*)(/[^ ]*)")}
        if #matches <= 0 then
            error("解析路径失败")
        end
        return matches[2], body
    end
    return "", body
end

---根据路径查找服务器对应的ip和请求方式对应的端口号
---@param path string
---@param serve string
---@return string ip
---@return integer port
ServeGetIpPort = function (path, serve)
    local ip = ""
    local port = 0
    if serve == "pytorch" then
        ip = "127.0.0.1"
        local matches = {lrex.match(path, "/([^#?&/]*)/([^#?&/]*)(/[^ ]*)")}
        if #matches <= 0 then
            error("解析路径失败")
        end
        if matches[2] == "inference" then
            port = 8080
        elseif matches[2] == "management" then
            port = 8081
        end
    elseif serve == "tensorflow" then
        ip = "124.221.123.251"
        port = 8501
    end
    return ip, port
end

--[==[
---将转化后的（方法，路径，载荷）转发到指定服务器的指定端口
---@param method string
---@param path string
---@param body string
---@param ip string
---@param port integer
---@return string response
RelayToServe = function (method, path, body, ip, port)
    print("RelayToServe")
    local serverHost = ip
    local serverPort = port
    local httpRequest  = ""
    if method == "GET" then
        httpRequest = method .. " " .. path .. " HTTP/1.1\r\nHost: " .. ip .. ":" .. port .. "\r\nAccept: */*" .. "\r\n\r\n"
    else
        httpRequest = method .. " " .. path .. " HTTP/1.1\r\nHost: " .. ip .. ":" .. port .. "\r\nAccept: */*\r\nContent-Length: " .. #body .. "\r\n\r\n" .. body
    end
    print("httpRequest:"..httpRequest)
    --[=[local client = socket.tcp()
    client:connect(serverHost, serverPort)
    client:settimeout(1)
    client:send(httpRequest)
    local response_line = ""
    local err = nil
    local response = ""
    print("..waiting")
    while response_line ~= nil do
        response = response .. response_line .. "\r\n"
        response_line, err = client:receive("*l")
    end
    print("..ok")
    if err then
        if err ~= "timeout" and err ~= "closed" then
            error("接受响应出现错误[" .. err .. "]")
        end
    end
    client:close()
    ]=]
    local response = libsocket(serverHost, serverPort, httpRequest)
    print(type(libsocket))
    print("response"..response)
    return response
end
]==]

---转发一个路径，并获取响应
---@param method string
---@param path string
---@param body string
---@return string response_body
---@return integer response_code
Relay = function (method, path, body)
    local _, server_list = PathGetServe(path)
    local converted_response_body = ""
    local converted_response_code = 200
    converted_response_body = "{\n  \"ServeList\":".."[\n    "
    for _, server in ipairs(server_list) do
        local converted_path, converted_body = PathBodyConvert(path, body, server)
        local ip, port = ServeGetIpPort(path, server)
        print("method:"..method)
        print("path:"..converted_path)
        print("bodylen:"..#body)
        print("body:"..body)
        print("ip:"..ip)
        print("port:"..port)
        local response_body, response_code = RelayToServe(method, converted_path, body, #body, ip, port)
        print("===============")
        print("response_body" .. response_body)
        print("===============")
        converted_response_body =
        converted_response_body .. "{\n      \"server\":\"" .. server .. "\",\n      " .. "\"code\": " .. response_code .. ",\n      " .. "\"response\":\n"

        for line in response_body:gmatch("[^\n]+") do
            converted_response_body = converted_response_body .. "        " .. line .. "\n"
        end
        converted_response_body = converted_response_body .. "    }\n"
    end
    converted_response_body = converted_response_body .. "  ]\n}\n"
    return converted_response_body, converted_response_code
end


--print(Relay("GET", "/pytorch/inference/ping", ""))
