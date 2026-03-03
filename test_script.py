import urllib.request
import json
import base64

try:
    # 1. Register
    req = urllib.request.Request('http://localhost:8000/users/', method='POST', headers={'Content-Type': 'application/json'}, data=b'{"username":"test2","email":"test2@test.com","password":"pass"}')
    urllib.request.urlopen(req)
except Exception as e:
    pass # might already exist

# 2. Login
req2 = urllib.request.Request('http://localhost:8000/token', method='POST', headers={'Content-Type': 'application/x-www-form-urlencoded'}, data=b'username=test2&password=pass')
res2 = json.loads(urllib.request.urlopen(req2).read())
token = res2['access_token']

# 3. Create project
data = {"name":"TestProj", "data":{"rooms":[], "house":{}}}
req3 = urllib.request.Request('http://localhost:8000/projects/', method='POST', headers={'Authorization': 'Bearer '+token, 'Content-Type': 'application/json'}, data=json.dumps(data).encode('utf-8'))
res3 = json.loads(urllib.request.urlopen(req3).read())
print("Created project:", res3)
