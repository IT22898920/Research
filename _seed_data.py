import json
import urllib.request
import random
import time

TOKEN = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjY5YWQxZWU5ODNlN2I3NzdhZWVjN2ViNCIsImlhdCI6MTc3ODAyNDY5NywiZXhwIjoxNzc4NjI5NDk3fQ.QQ4SQwb1pEPFslfvYn2o9fR-R9QiJ9_Ea7lCUJwMmIA'
BASE = 'https://research-production-ed2e.up.railway.app/api'
PLANTATION_ID = '69f6e1bf9c9e7a59302081ce'

def api(method, path, body=None):
    url = BASE + path
    headers = {'Authorization': 'Bearer ' + TOKEN, 'Content-Type': 'application/json'}
    data = json.dumps(body).encode('utf-8') if body else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        return {'error': str(e), 'success': False}

res = api('GET', f'/trees/plantation/{PLANTATION_ID}')
trees = res.get('data', [])
print(f'Found {len(trees)} trees')

scenarios = [
    {'status':'healthy','scanType':'all','label':'All Healthy','msg':'No pests detected.','conf':0.95,'issues':[]},
    {'status':'healthy','scanType':'all','label':'All Healthy','msg':'No pests detected.','conf':0.93,'issues':[]},
    {'status':'healthy','scanType':'all','label':'All Healthy','msg':'No pests detected.','conf':0.96,'issues':[]},
    {'status':'healthy','scanType':'all','label':'All Healthy','msg':'No pests detected.','conf':0.94,'issues':[]},
    {'status':'healthy','scanType':'all','label':'All Healthy','msg':'No pests detected.','conf':0.92,'issues':[]},
    {'status':'healthy','scanType':'all','label':'All Healthy','msg':'No pests detected.','conf':0.97,'issues':[]},
    {'status':'healthy','scanType':'all','label':'All Healthy','msg':'No pests detected.','conf':0.94,'issues':[]},
    {'status':'healthy','scanType':'all','label':'All Healthy','msg':'No pests detected.','conf':0.95,'issues':[]},
    {'status':'infected','scanType':'mite','label':'Coconut Mite Infected','msg':'Mite damage on coconut surface','conf':0.92,'issues':['coconut_mite detected']},
    {'status':'infected','scanType':'mite','label':'Coconut Mite Infected','msg':'Severe mite infestation','conf':0.87,'issues':['coconut_mite detected']},
    {'status':'infected','scanType':'mite','label':'Coconut Mite Infected','msg':'Early stage mite damage','conf':0.78,'issues':['coconut_mite detected']},
    {'status':'infected','scanType':'caterpillar','label':'Caterpillar Damage','msg':'Black-headed caterpillar on leaves','conf':0.94,'issues':['caterpillar detected']},
    {'status':'infected','scanType':'caterpillar','label':'Caterpillar Damage','msg':'Caterpillar damage on young leaves','conf':0.86,'issues':['caterpillar detected']},
    {'status':'infected','scanType':'white_fly','label':'White Fly Infestation','msg':'White fly colonies on leaf undersides','conf':0.91,'issues':['white_fly detected']},
    {'status':'infected','scanType':'white_fly','label':'White Fly Infestation','msg':'Moderate white fly infestation','conf':0.83,'issues':['white_fly detected']},
    {'status':'infected','scanType':'disease','label':'Leaf Rot','msg':'Leaf rot disease detected','conf':0.95,'issues':['Leaf Rot disease detected']},
    {'status':'infected','scanType':'disease','label':'Leaf Rot','msg':'Advanced leaf rot symptoms','conf':0.88,'issues':['Leaf Rot disease detected']},
    {'status':'infected','scanType':'disease','label':'Leaf Spot','msg':'Leaf spot disease symptoms','conf':0.89,'issues':['Leaf_Spot disease detected']},
    {'status':'unhealthy','scanType':'leaf_health','label':'Unhealthy Leaf','msg':'Nutrient deficiencies detected','conf':0.81,'issues':['Unhealthy leaf detected','Nitrogen Deficiency','Potassium Deficiency']},
    {'status':'unhealthy','scanType':'leaf_health','label':'Unhealthy Leaf','msg':'Magnesium and water stress','conf':0.79,'issues':['Unhealthy leaf detected','Magnesium Deficiency','Water Stress (Under-watering)']},
]

for i, tree in enumerate(trees):
    if i >= len(scenarios):
        break
    sc = scenarios[i]
    tree_id = tree['_id']
    label = tree['label']

    # Realistic: Healthy tree has 6-8 mature bunches (4+ months), each bunch ~15-25 nuts
    if sc['status'] == 'healthy':
        bunches = random.randint(6, 8)
        nuts = bunches * random.randint(15, 25)  # 90-200 nuts
        pests = 0
    elif sc['status'] == 'unhealthy':
        bunches = random.randint(3, 5)
        nuts = bunches * random.randint(10, 18)  # 30-90 nuts
        pests = random.randint(2, 5)
    else:  # infected
        bunches = random.randint(2, 4)
        nuts = bunches * random.randint(8, 15)   # 16-60 nuts
        pests = random.randint(8, 25)

    r1 = api('PUT', f'/trees/{tree_id}/scan-results', {
        'nutCount': nuts, 'bunchCount': bunches, 'pestCount': pests,
        'detectedIssues': sc['issues'], 'healthStatus': sc['status'],
    })
    r2 = api('POST', f'/trees/{tree_id}/health-scan', {
        'status': sc['status'], 'scanType': sc['scanType'], 'confidence': sc['conf'],
        'details': {'prediction': {'label': sc['label'], 'confidence': sc['conf'], 'message': sc['msg']}},
    })

    s1 = 'OK' if r1.get('success') else 'FAIL'
    s2 = 'OK' if r2.get('success') else 'FAIL'
    print(f"  {label}: {sc['label']:30s} | nuts:{nuts:3d} bunches:{bunches:2d} pests:{pests:2d} | scan:{s1} hist:{s2}")
    time.sleep(0.3)

print('DONE!')
