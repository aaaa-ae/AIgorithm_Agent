import sys
import requests
import re
import json
import base64

def login(name, password):
    # 正则表达式匹配json数据
    pattern = re.compile(r'\{.*\}')

    try:
        # 请求检查当前状态
        r = requests.get('https://w.seu.edu.cn/drcom/chkstatus?callback=dr1002')
    except requests.exceptions.RequestException as e:
        print(f'错误：连接失败。详细信息：{str(e)}')
        return False

    if r.status_code != 200:
        print(f'错误：连接失败，状态码：{r.status_code}')
        return False
    
    # 解析返回的json数据
    try:
        status = json.loads(pattern.findall(r.text)[0])
    except json.JSONDecodeError:
        print('错误：解析响应数据失败。')
        return False

    if status['result'] == 1:
        print('错误：你已经登录了 seu-wlan。')
        return False
    elif status['result'] != 0:
        print(f'错误：未知错误，返回状态：{status["result"]}')
        return False

    # 登录信息
    config = {}
    config['username'] = name
    config['password'] = password

    login_url = f'https://w.seu.edu.cn:801/eportal/?c=Portal&a=login&callback=dr1003&login_method=1&user_account=%2C0%2C{config["username"]}&user_password={config["password"]}&wlan_user_ip={status["v46ip"]}'
    
    try:
        r = requests.get(login_url)
    except requests.exceptions.RequestException as e:
        print(f'错误：连接失败。详细信息：{str(e)}')
        return False

    if r.status_code != 200:
        print(f'错误：连接失败，状态码：{r.status_code}')
        return False

    # 解析登录响应数据
    try:
        login = json.loads(pattern.findall(r.text)[0])
    except json.JSONDecodeError:
        print('错误：解析登录响应数据失败。')
        return False

    if login['result'] != '1':
        message = base64.b64decode(login['msg']).decode()
        if message == 'ldap auth error':
            print('错误：用户名或密码错误。')
        else:
            print(f'错误：登录失败，错误消息：{message}')
        return False

    print('登录成功！')
    return True


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('错误：请提供用户名和密码。')
    else:
        login(sys.argv[1], sys.argv[2])
