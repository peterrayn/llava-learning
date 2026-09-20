# %%
# python [epochs] [batch_size] [Dataset_dir]
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from pathlib import Path
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device=torch.device('cpu')
torch.manual_seed(123)
print(torch.cuda.get_device_name(0))

# %%
import sys
if 'ipykernel' in sys.argv[0]:
    EPOCHS=1
    BATCH_SIZE=256
    Dataset_dir='/data/hujun'
else:
    arg_len=len(sys.argv)
    if arg_len!=4:
        print('should be [epochs] [batch_size] [Dataset_dir]')
        sys.exit()
    EPOCHS=int(sys.argv[1])
    BATCH_SIZE=int(sys.argv[2])
    Dataset_dir=str(sys.argv[3])
    if not Path(Dataset_dir).is_dir():
        print('worng dir: ',Dataset_dir)
        sys.exit()

# %%
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers import CLIPImageProcessor, CLIPVisionModel
from PIL import Image
import requests
import matplotlib.pyplot as plt


demo_vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer=AutoTokenizer.from_pretrained('distilgpt2')
demo_gpt2=AutoModelForCausalLM.from_pretrained('distilgpt2')

tokenizer.pad_token=tokenizer.eos_token

# %%

image = Image.open('./kitty.jpg')
inputs = processor(images=image, return_tensors="pt")
outputs = demo_vision_encoder(**inputs)

outputs.last_hidden_state.shape

# %%
from datasets import load_dataset
caption_dataset=load_dataset(Dataset_dir+'/flickr8k')['train']
v_dataset=load_dataset(Dataset_dir+'/vqav2-small')['validation']
caption_test_dataset=load_dataset(Dataset_dir+'/flickr8k')['test']
split=v_dataset.train_test_split(test_size=0.2,seed=42)
vqa_dataset=split['train']
vqa_test_dataset=split['test']

# %%
print(caption_dataset[0])
print(vqa_dataset[0])

# %%
import random
import re

def collate_fn(batch):
    # [{img,caption0-4}---{img,caption0-4}]
    img=[b['image'] for b in batch]
    img=processor(img,return_tensors='pt')['pixel_values']
    prompt="Describe the image:"
    text=[prompt+" "+b['caption_0']+'<|endoftext|>' for b in batch]
    text=tokenizer.encode(text,padding=True,return_tensors='pt')
    eos_len=[sum(t==tokenizer.eos_token_id).item() for t in text]
    in_text=text[:,:-1]
    out_text=text[:,1:].clone()
    # print(out_text[0])
    out_text[:,:len(tokenizer.encode(prompt))-1]=-100
    for i in range(out_text.shape[0]):
        if eos_len[i]>1:
            out_text[i,-(eos_len[i]-1):]=-100
    return img,in_text,out_text
caption_dataloader=DataLoader(dataset=caption_dataset,batch_size=BATCH_SIZE,collate_fn=collate_fn,shuffle=True)

# %%
def vqa_collate_fn(batch):
    # [{'multiple_choice_answer','question','image'}---]
    img=[b['image'] for b in batch]
    img=processor(img,return_tensors='pt')['pixel_values']
    text=[b['question']+" "+b['multiple_choice_answer']+'<|endoftext|>' for b in batch]
    # print('text',text)
    question_len=[len(tokenizer.encode(b['question'])) for b in batch]
    text=tokenizer.encode(text,padding=True,return_tensors='pt')
    eos_len=[sum(t==tokenizer.eos_token_id).item() for t in text]
    in_text=text[:,:-1]
    out_text=text[:,1:].clone()
    # print(out_text[0])
    for i in range(out_text.shape[0]):
        out_text[i,:(question_len[i]-1)]=-100
        if eos_len[i]>1:
            out_text[i,-(eos_len[i]-1):]=-100
    return img,in_text,out_text
vqa_dataloader=DataLoader(dataset=vqa_dataset,batch_size=BATCH_SIZE,collate_fn=vqa_collate_fn,shuffle=True)

# %%
for img,in_text,out_text in vqa_dataloader:
    print(img.shape)
    print(tokenizer.decode(in_text[2]))
    print(in_text[2])
    print(out_text[2])
    break

# %%
for img,in_text,out_text in caption_dataloader:
    print(img.shape)
    print(in_text.shape)
    print(tokenizer.decode(in_text[3]))
    print(in_text[3])
    print(out_text[3])
    # print(in_text[2])
    # print([{i.item():tokenizer.convert_ids_to_tokens(i.item())} for i in out_text[0]])
    # print(out_text[1])
    # print(out_text[2])
    # print(tokenizer.decode(in_text[0]))
    # print(tokenizer.decode(out_text[0]))
    break

# %%
class Projector(nn.Module):
    def __init__(self,input_dim,out_dim):
        super().__init__()
        self.linear=nn.Linear(input_dim,out_dim)
    def forward(self,x):
        return self.linear(x)


# %%
class Llava(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder=CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.gpt2=AutoModelForCausalLM.from_pretrained('distilgpt2').to(device)
        self.projector=Projector(768,768).to(device)
        # set false
        self.llm_embed_token=self.gpt2.get_input_embeddings()
    def forward(self,img,text):
        with torch.no_grad():
            img_embed=self.vision_encoder(img).last_hidden_state[:,1:,:]
        img_embed=self.projector(img_embed)
        text_embed=self.llm_embed_token(text)
        # print(img_embed.shape,text_embed.shape)
        combind_embed=torch.concatenate([img_embed,text_embed],dim=1)
        out=self.gpt2(inputs_embeds=combind_embed).logits
        return out


# %%
# show the stupid effect

def see_effect(image,model,length=50,prompt='Describe the image:'):
# image = Image.open('./kitty.jpg')
    plt.imshow(image)
    input_img = processor(images=image, return_tensors="pt").pixel_values.to(device)
    input_text=tokenizer.encode(prompt,return_tensors='pt').to(device)
    model.eval()
    with torch.no_grad():
        for i in range(length):
            out=model(input_img,input_text)
            predict_token=torch.argmax(out[:,-1:,:],dim=-1)
            input_text=torch.concatenate([input_text,predict_token],dim=-1)
            if predict_token==tokenizer.eos_token_id:
                break
    print(tokenizer.decode(input_text))

# %%
def train(model,dataloader,optimizer,epochs,criterion):
    L=[]
    for epoch in range(epochs):
        for i,(img,in_text,out_text) in enumerate(dataloader):
            img,in_text,out_text=img.to(device),in_text.to(device),out_text.to(device)
            out=model(img,in_text)
            loss=criterion(out[:,-out_text.shape[1]:,:].reshape(-1,tokenizer.vocab_size),out_text.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            L.append(loss.item())
            # if i>10:
            #     break
            print(f"{epoch} {i}/{len(dataloader)}",loss.item())
    return L


# %%
# stage 1
def stage1(model):
    print("=========stage1========")
    for para in model.vision_encoder.parameters():
        para.requires_grad=False
    for para in model.gpt2.parameters():
        para.requires_grad=False
    criterion=nn.CrossEntropyLoss()
    optimizer1=torch.optim.AdamW(model.projector.parameters(),lr=0.001)
    model.eval()
    model.projector.train()
    stage1_loss=train(model=model,dataloader=caption_dataloader,optimizer=optimizer1,epochs=EPOCHS,criterion=criterion)
    return stage1_loss

# %%
# stage 2
def stage2(model):
    print("=========stage2========")
    for para in model.vision_encoder.parameters():
        para.requires_grad=False
    for para in model.gpt2.parameters():
        para.requires_grad=True
    criterion=nn.CrossEntropyLoss()
    optimizer2 = torch.optim.AdamW([
        {"params": model.projector.parameters(), "lr": 1e-4},
        {"params": model.gpt2.parameters(), "lr": 1e-5},
    ])
    model.eval()
    model.projector.train()
    model.gpt2.train()
    stage2_loss=train(model=model,dataloader=vqa_dataloader,optimizer=optimizer2,epochs=EPOCHS,criterion=criterion)
    return stage2_loss

# %%
# start experiment
###
# model0,model1,model2,model3
# 
# 
###

model0,model1,model2,model3=Llava().to(device),Llava().to(device),Llava().to(device),Llava().to(device)
model1.load_state_dict(model0.state_dict())
model2.load_state_dict(model0.state_dict())
torch.save({'state_dict':model0.state_dict(),'loss':[]},'model_notrain.ckpt')
model1_loss=stage1(model1)
torch.save({'state_dict':model1.state_dict(),'loss':model1_loss},'model_stage1.ckpt')
model2_loss=stage2(model2)
torch.save({'state_dict':model2.state_dict(),'loss':model2_loss},'model_stage2.ckpt')
model3.load_state_dict(model1.state_dict())
model3_loss=stage2(model3)
torch.save({'state_dict':model3.state_dict(),'loss':model3_loss},'model_stage12.ckpt')

# %%
img=Image.open('./kitty.jpg')
prompt='What is it? cat. How many? 2. What is the color of the cat?'
print('Question: ',prompt)
see_effect(img,prompt=prompt,model=model0)
see_effect(img,prompt=prompt,model=model1)
see_effect(img,prompt=prompt,model=model2)


