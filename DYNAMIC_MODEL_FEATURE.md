# Dinamik Model Entegrasyonu Özelliği

Bu belge, LLM entegrasyon sistemine eklenen dinamik model yükleme özelliğini açıklamaktadır.

## Genel Bakış

Bu özellik, sisteme yeni modellerin kolaylıkla eklenebilmesini ve mevcut modellerin dinamik olarak yapılandırılabilmesini sağlar. Kullanıcılar, modelleri sadece kod değişikliği gerektirmeden, yapılandırma dosyaları aracılığıyla tanımlayabilir, ekleyebilir ve güncelleyebilirler.

## Özellikler

1. **Yapılandırma Dosyaları ile Model Tanımlama**: Modeller YAML formatında bir yapılandırma dosyasında tanımlanabilir.
2. **Ortam Değişkenleri Desteği**: API anahtarları, ortam değişkenlerinden referans alınarak güvenli bir şekilde saklanabilir.
3. **Yerel Model Keşfi**: Ollama gibi yerel model sunucularındaki mevcut modeller otomatik olarak keşfedilebilir.
4. **Model Kayıt Sistemi**: Yeni model türleri, merkezi bir kayıt sistemi aracılığıyla eklenebilir.
5. **Komut Satırı Aracı**: Modelleri komut satırından eklemek ve listelemek için bir araç (`add_model.py`).

## Kullanım Örnekleri

### 1. Modelleri Yapılandırma Dosyasından Yükleme

```python
from models import ModelManager
from agent import Agent

# Konfigürasyon dosyasından modelleri yükleme
model_manager = ModelManager()
model_manager.load_from_config("models_config.yaml")

# Ek olarak, yerel modelleri keşfetme
model_manager.discover_ollama_models()
```

### 2. Yeni Model Ekleme (Komut Satırı)

```bash
# Model listeleme
python add_model.py list

# Yeni model ekleme
python add_model.py add --type ollama --model llama3.1:8b --name llama3-custom --temperature 0.8 --num-ctx 8192

# API anahtarıyla model ekleme
python add_model.py add --type openai --model gpt-4o --name creative-gpt4 --api-key "your_api_key" --temperature 1.0
```

### 3. Farklı Görevler için Modelleri Dinamik Eşleştirme

```python
# Kullanılabilir modelleri listeleme
models = model_manager.list_models()

# Belirli model türlerini arama
openai_models = [m for m in models if m.startswith("openai_") or m.startswith("gpt")]
ollama_models = [m for m in models if m.startswith("ollama_")]

# Görev türleri için modelleri dinamik atama
for task_type, model_prefix in [
    ("summarization", "ollama_llama3"),
    ("sentiment_analysis", "claude"),
    ("information_extraction", "gpt4"),
]:
    matching_models = [m for m in models if m.startswith(model_prefix)]
    if matching_models:
        agent.set_default_model(task_type, matching_models[0])
```

## Model Yapılandırma Formatı

```yaml
models:
  - type: openai
    name: gpt4-turbo
    api_key: env:OPENAI_API_KEY  # Ortam değişkeninden yükle
    model_name: gpt-4-turbo-2024-04-09
    temperature: 0.7
    max_tokens: 1500
    
  - type: ollama
    name: llama3-custom
    model_name: llama3.1:8b
    temperature: 0.8
    num_ctx: 8192
    
  - type: anthropic
    name: claude-opus
    api_key: env:ANTHROPIC_API_KEY
    model_name: claude-3-opus-20240229
    temperature: 0.7
    max_tokens: 2000
```

## Desteklenen Model Türleri

| Model Türü  | Açıklama                                  | Parametreler                                |
|-------------|-------------------------------------------|---------------------------------------------|
| `openai`    | OpenAI API modelleri (GPT-3.5, GPT-4)     | api_key, model_name, temperature, max_tokens |
| `anthropic` | Anthropic Claude modelleri                | api_key, model_name, temperature, max_tokens |
| `gemini`    | Google Gemini modelleri                   | api_key, model_name, temperature            |
| `mistral`   | Mistral AI modelleri                      | api_key, model_name, temperature, max_tokens |
| `deepseek`  | DeepSeek V2 modelleri                     | api_key, model_name, temperature, max_tokens |
| `phi4`      | Microsoft Phi-4 modelleri                 | api_key, model_name                         |
| `ollama`    | Yerel Ollama modelleri                    | model_name, temperature, num_ctx            |
| `dummy`     | Test için yapay model                     | (parametre yok)                             |

## Yeni Model Türü Ekleme

```python
from models import LLMModel, ModelRegistry

# Yeni model sınıfı tanımlama
class YeniModelAPI(LLMModel):
    def __init__(self, api_key, model_name, name=None, temperature=0.7):
        super().__init__(name or f"yeni_{model_name}")
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        
    def generate(self, prompt):
        # API çağrısı yapan kodu buraya yazın
        return "API yanıtı burada"

# Sınıfı model kayıt sistemine ekleme
ModelRegistry.register_model_class('yeni_api', YeniModelAPI)
```

## Güvenlik Uygulamaları

- API anahtarları doğrudan yapılandırma dosyasında saklanmak yerine, ortam değişkenlerinden yüklenir.
- Hassas bilgileri içeren ortam değişkenleri, `env:VARIABLE_NAME` sözdizimi kullanılarak belirtilir.
- API anahtarları komut satırı aracında kullanıldığında, otomatik olarak ortam değişkenlerine dönüştürülür.

## Potansiyel Geliştirmeler

1. **Web Arayüzü**: Model yönetimi için bir web arayüzü eklenebilir.
2. **Model Performans İzleme**: Modellerin yanıt süreleri ve başarı oranları izlenebilir.
3. **Otomatik Yedekleme**: Modellerin yedekleme mekanizması, bir model başarısız olduğunda alternatif modele geçiş yapabilir.
4. **Model Havuzu**: Aynı görev için birden fazla model kullanarak sonuçların birleştirilmesi sağlanabilir.
5. **Dinamik İstem Optimizasyonu**: Her model için optimize edilmiş sistem istemleri otomatik oluşturulabilir. 