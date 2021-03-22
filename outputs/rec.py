l1="education standard"
l2="income level"

previousx= ""
previousy= ""


def split(l1,l2,previousx,previousy):
    l1_split=l1.split(" ")
    l2_split=l2.split(" ")
    if len(l1_split)>1 or len(l2_split)>1:
        for x in range(len(l1_split)):
            previousx = ""
            previousy=""
            for y in range(len(l2_split)):
                ret=split(l1_split[x],l2_split[y],previousx,previousy)
                previousy = l2_split[y]
                previousx = l1_split[x]
                print(ret)
    else:
        return previousx+" "+l1+ " "+previousy +" "+l2


split(l1, l2, previousx,previousy)