from google import genai
client = genai.Client(api_key='AIzaSyCoSY5GuS30R9SKFEBiPiv5NWr7DXi3vLk')
for model in client.models.list():
    print(model.name)
