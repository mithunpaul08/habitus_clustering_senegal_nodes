import json
input_tokens=["education","stability"]
d={
    "PROMOTES": 0.0005161457811482251,
    "DOES_NOT_PROMOTE": 0.0036508317571133375,

}

d2={
    "INHIBITS": 0.0004498149792198092,
    "DOES_NOT_INHIBIT": 0.0018296359339728951,
}

def convert_probability_dict_to_pretty_print(data,input_tokens,pretty_dict):
    for k, v in data.items():
        pretty_key = f"{input_tokens[0]} {k} {input_tokens[1]}"
        pretty_dict[pretty_key] = v
    return pretty_dict

pretty_dict={}
convert_probability_dict_to_pretty_print(d,input_tokens,pretty_dict)
convert_probability_dict_to_pretty_print(d2,input_tokens,pretty_dict)



str_json=json.dumps(pretty_dict,indent=4)
with open("temp.json","w") as f:
	f.write(str_json)
