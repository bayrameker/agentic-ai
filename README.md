# Çoklu LLM API Entegrasyonu için Modüler Python Eklentisi

Bu proje, farklı Büyük Dil Modellerini (LLM) tek bir arayüz üzerinden kullanmanızı sağlayan modüler ve genişletilebilir bir Python eklentisidir. OpenAI, Anthropic Claude, Google Gemini, Mistral AI, DeepSeek ve yerel Ollama modelleri gibi çeşitli API'lerle çalışabilen bu sistem, metin özetleme, duygu analizi, bilgi çıkarımı gibi farklı görevleri dinamik şekilde yürütebilir.

## Mimari

Proje iki ana bileşenden oluşmaktadır:

1. **Agentic Yapı (Agent)**: Kullanıcı taleplerini analiz ederek görevlere bölen, bu görevleri kuyruğa alan ve uygun modele yönlendiren akıllı ajan sistemi.
2. **LLM Sorgulama Modülü (Models)**: Farklı LLM API'lerine bağlanan, her görev için uygun sistem istemleri (system prompts) ile modelleri çağıran ve sonuçları döndüren modül.

## Özellikler

- **Modüler Tasarım**: Yeni modeller kolayca eklenebilir veya mevcut modeller güncellenebilir
- **Dinamik Model Entegrasyonu**: Modelleri konfigürasyon dosyası ile tanımlama ve dinamik olarak yükleme
- **Çoklu API Desteği**: Farklı yapıdaki API'lere erişmek için ortak bir arayüz
- **Yerel Model Keşfi**: Ollama gibi yerel modelleri otomatik keşfetme ve kullanma
- **Görev Kuyruğu Yönetimi**: Görevleri planlama, önceliklendirme ve yürütme
- **Doğal Dil İşleme**: Kullanıcı isteklerinden görevleri otomatik çıkarabilme yeteneği
- **Farklı Çıktı Formatları**: Sonuçlar JSON, YAML veya düz metin olarak alınabilir
- **Kolay Entegrasyon**: Web servisi veya kütüphane olarak kullanılabilir

## Desteklenen Modeller

### API Tabanlı Modeller
- **OpenAI**: GPT-3.5 Turbo, GPT-4 Turbo, GPT-4o
- **Anthropic**: Claude 3 Opus, Claude 3 Sonnet, Claude 3 Haiku
- **Google**: Gemini 1.5 Pro
- **Mistral AI**: Mistral Large
- **DeepSeek**: DeepSeek V2 (128k context window)
- **Microsoft**: Phi-4

### Yerel Modeller (Ollama ile)
- **Llama 3**: 70B, 8B
- **Mixtral**: 8x22B, 8x7B
- **DeepSeek**: 67B, 33B
- **Phi-4**: 3.8B, 7B
- Ve diğer Ollama destekli modeller

## Kurulum

```bash
git clone https://github.com/kullanici/llm-integration-plugin.git
cd llm-integration-plugin
pip install -r requirements.txt

# Eğer belirli API'leri kullanacaksanız, requirements.txt içindeki
# ilgili kütüphane satırlarını uncomment edin.
```

## Kullanım

### 1. Basit Örnek

```python
from models import ModelManager, DummyModel
from agent import Agent
from tasks import Task

# Model yöneticisi ve modellerin ayarlanması
model_manager = ModelManager()
model_manager.add_model("default_model", DummyModel("test_model"))

# Ajanın oluşturulması
agent = Agent(model_manager)

# Görev ekleme
agent.add_task(Task("summarization", "Özetlenecek metin burada."))

# Görevlerin çalıştırılması
results = agent.run_all_tasks()
print(results)
```

### 2. Konfigürasyon Dosyasından Model Yükleme

Modelleri `models_config.yaml` dosyasında tanımlayabilirsiniz:

```yaml
# models_config.yaml
models:
  - type: openai
    name: gpt4-turbo
    api_key: env:OPENAI_API_KEY  # Ortam değişkeninden yükle
    model_name: gpt-4-turbo-2024-04-09
    temperature: 0.7
    max_tokens: 1500
    
  - type: ollama
    name: llama3-8b
    model_name: llama3:8b
    temperature: 0.7
    num_ctx: 4096
```

Konfigürasyondan modelleri yüklemek için:

```python
from models import ModelManager
from agent import Agent
from tasks import Task

# Konfigürasyon dosyasından modelleri yükleme
model_manager = ModelManager()
model_manager.load_from_config("models_config.yaml")

# Ajan oluşturma ve görev ekleme
agent = Agent(model_manager)
agent.add_task(Task("summarization", "Konfigürasyondan yüklenen modelle özetlenecek metin."))

# Çalıştırma
results = agent.run_all_tasks()
print(results)
```

### 3. Dinamik Model Ekleme

Ayrıca, eklediğimiz `add_model.py` betiği ile dinamik olarak model ekleyebilirsiniz:

```bash
# Mevcut modelleri listeleme
python add_model.py list

# Ollama modeli ekleme
python add_model.py add --type ollama --model llama3:8b --name llama3-mini --temperature 0.8 --num-ctx 8192

# OpenAI modeli ekleme
python add_model.py add --type openai --model gpt-4o --name gpt4o-creative --api-key "your_api_key" --temperature 1.0
```

### 4. OpenAI API Kullanımı

```python
import os
from models import ModelManager, OpenAIModel
from agent import Agent
from tasks import Task

# API anahtarını ortam değişkeninden alma
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Model yöneticisi oluşturma
model_manager = ModelManager()
model_manager.add_model(
    "gpt4", 
    OpenAIModel(
        api_key=openai_api_key, 
        model_name="gpt-4-turbo-2024-04-09",
        temperature=0.7, 
        max_tokens=1500
    )
)

# Ajan oluşturma ve görev ekleme
agent = Agent(model_manager)
agent.set_default_model("summarization", "gpt4")
agent.add_task(Task("summarization", "OpenAI GPT-4 modeliyle özetlenecek uzun bir metin."))

# Çalıştırma ve sonucu alma
results = agent.run_all_tasks(output_format="json")
print(results)
```

### 5. Ollama ile Yerel Modellerin Kullanımı

```python
from models import ModelManager, OllamaModel
from agent import Agent
from tasks import Task

# Model yöneticisi oluşturma
model_manager = ModelManager()

# Ollama modellerini ekleme (Ollama servisinin çalışıyor olması gerekir)
model_manager.add_model(
    "llama3", 
    OllamaModel(
        model_name="llama3:70b",
        server_url="http://localhost:11434",
        temperature=0.7,
        num_ctx=4096
    )
)

# Ajan oluşturma ve görev ekleme
agent = Agent(model_manager)
agent.set_default_model("sentiment_analysis", "llama3")
agent.add_task(Task("sentiment_analysis", "Bu ürün beklentilerimi fazlasıyla karşıladı ve çok memnunum."))

# Çalıştırma
results = agent.run_all_tasks()
print(results)
```

### 6. Yerel Ollama Modellerini Otomatik Keşfetme

```python
from models import ModelManager
from agent import Agent
from tasks import Task

# Model yöneticisi oluşturma ve yerel modelleri keşfetme
model_manager = ModelManager()
ollama_models = model_manager.discover_ollama_models()

if ollama_models:
    print(f"Bulunan Ollama modelleri: {', '.join(ollama_models)}")
    
    # Ajan oluşturma
    agent = Agent(model_manager)
    
    # İlk bulunan modeli kullanma
    model_name = f"ollama_{ollama_models[0]}"
    agent.set_default_model("summarization", model_name)
    
    # Görev ekleme ve çalıştırma
    agent.add_task(Task("summarization", "Otomatik keşfedilen yerel modelle özetlenecek metin."))
    results = agent.run_all_tasks()
    print(results)
```

### 7. Claude, Gemini ve Diğer API'ler

```python
import os
from models import ModelManager, AnthropicModel, GoogleGeminiModel
from agent import Agent
from tasks import Task

# API anahtarlarını alma
anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
google_key = os.environ.get("GOOGLE_API_KEY")

# Model yöneticisi oluşturma
model_manager = ModelManager()

# Modelleri ekleme
model_manager.add_model(
    "claude", 
    AnthropicModel(
        api_key=anthropic_key, 
        model_name="claude-3-opus-20240229"
    )
)

model_manager.add_model(
    "gemini", 
    GoogleGeminiModel(
        api_key=google_key, 
        model_name="gemini-1.5-pro-latest"
    )
)

# Ajan oluşturma ve farklı görevler için farklı modeller atama
agent = Agent(model_manager)
agent.set_default_model("summarization", "claude")
agent.set_default_model("information_extraction", "gemini")

# Görevler ekleyip çalıştırma
agent.add_task(Task("summarization", "Claude ile özetlenecek metin."))
agent.add_task(Task("information_extraction", "Gemini ile bilgi çıkarılacak metin."))
results = agent.run_all_tasks(output_format="text")
print(results)
```

### 8. Kullanıcı İsteğinden Görev Çıkarma

```python
# Kullanıcı isteğinden görevlerin otomatik çıkarılması
kullanici_istegi = "Bu metni özetle ve duygu analizini yap: Bugün hava çok güzel."
agent.plan_tasks_from_request(kullanici_istegi)

# Çıkarılan görevlerin yürütülmesi
results = agent.run_all_tasks(output_format="json")
print(results)
```

## Örnek Çalıştırma

Tüm örnekleri görmek için:

```bash
python main.py
```

Belirli bir örneği çalıştırmak için:

```bash
python main.py --example parsing     # Kullanıcı isteği analizi örneği
python main.py --example json        # JSON formatında çıktı örneği
python main.py --example models      # Farklı modellerin atanması örneği
python main.py --example ollama      # Ollama ile yerel model kullanım örneği
python main.py --example dynamic     # Dinamik model yükleme örneği
python main.py --example api         # API benzeri kullanım simülasyonu
```

Belirli bir konfigürasyon dosyasıyla çalıştırma:

```bash
python main.py --config my_models.yaml
```

## Görev Türleri

Sistemin desteklediği görev türleri:

- `summarization`: Metin özetleme
- `sentiment_analysis`: Duygu analizi
- `information_extraction`: Metinden bilgi çıkarımı
- `translation`: Çeviri
- `code_generation`: Kod oluşturma
- `question_answering`: Soru yanıtlama

## Genişletme

### Yeni Bir Model Ekleme

#### 1. Kod ile Ekleme

Yeni bir LLM API'sini entegre etmek için `LLMModel` sınıfından türetilen yeni bir sınıf oluşturun:

```python
from models import LLMModel, ModelRegistry

class YeniModelAPI(LLMModel):
    def __init__(self, api_key, model_name, name=None):
        super().__init__(name or f"yeni_{model_name}")
        self.api_key = api_key
        self.model_name = model_name
        
    def generate(self, prompt):
        # API çağrısı yapan kodu buraya yazın
        return "API yanıtı buraya"

# Sınıfı model kayıt sistemine ekleyin
ModelRegistry.register_model_class('yeni_api', YeniModelAPI)

# Modeli sisteme ekleme
model_manager.add_model("yeni_model", YeniModelAPI(api_key="...", model_name="..."))
```

#### 2. Konfigürasyon ile Ekleme

Önce model sınıfını kaydedin, ardından konfigürasyon dosyasına ekleyin:

```yaml
# models_config.yaml
models:
  # Mevcut modeller...
  
  # Yeni model
  - type: yeni_api
    name: yeni-model
    api_key: env:YENI_API_KEY
    model_name: model-v1
```

#### 3. add_model.py ile Ekleme

```bash
python add_model.py add --type yeni_api --model model-v1 --name yeni-model --api-key "your_api_key"
```

### Yeni Bir Görev Türü Ekleme

Yeni bir görev türü için:

1. Ajan içinde yeni görev için sistem istemi tanımlayın:
```python
agent.system_prompts["yeni_gorev"] = "Yeni görev için sistem istemi"
```

2. Varsayılan model atayın:
```python
agent.default_model_for_task["yeni_gorev"] = "uygun_model_adi"
```

3. `plan_tasks_from_request` metoduna yeni görevin tanınması için gerekli kodu ekleyin

## API Anahtarları ve Güvenlik

API anahtarlarını doğrudan kod içine yazmak yerine, ortam değişkenleri veya `.env` dosyası ile yönetin:

```python
# .env dosyası kullanımı
from dotenv import load_dotenv
load_dotenv()  # .env dosyasını yükler

# Ortam değişkenlerinden anahtarları alma
openai_api_key = os.environ.get("OPENAI_API_KEY")
anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
```

Konfigürasyon dosyasında API anahtarını ortam değişkeninden referans ile kullanma:

```yaml
models:
  - type: openai
    name: gpt4
    api_key: env:OPENAI_API_KEY  # OPENAI_API_KEY ortam değişkeninden alınır
    model_name: gpt-4-turbo
```

## Lisans

MIT

