# Runbook do fork pessoal

Notas de manutenção deste checkout. **Não é** o guia de arquitetura do
projeto: esse é o [`AGENTS.md`](AGENTS.md) do `crmne/zapfast`, que é
intocado. Este arquivo é só sobre como manter o fork, sincronizar com o
upstream e instalar a build local na máquina.

> Por que outro arquivo? O `AGENTS.md` da raiz existe no upstream. Criar um
> arquivo com o mesmo nome aqui daria conflito em **todo** rebase futuro.
> `AGENTS-fork.md` não existe no upstream, então nunca conflita.

## O que este fork é

O ZapFast acrescido de uma extensão de transcrição local de mensagens de voz,
usando `whisper-rs` in-process, só CPU. O botão fica na bolha de áudio, o texto
aparece dentro dela, e os clipes ficam só na memória da sessão.

| | |
| --- | --- |
| Branch de trabalho | `transcription` |
| Remoto do dono | `upstream` → `https://github.com/crmne/zapfast.git` |
| Remoto próprio | `origin` → `https://github.com/matheuswidder/zapfast.git` |
| Fork | <https://github.com/matheuswidder/zapfast> |
| Base neste momento | `upstream/main` @ `df01459` |

Três commits, deliberadamente separados para deixar o rebase barato:

1. `Add local Whisper transcription for voice messages` — todo o código
2. `Show transcription states in the headless layout tests` — `src/demo.rs`
3. `Document the Transcription extension` — `docs/` e `assets/i18n/pt-BR.po`

A branch está no GitHub. **Se a máquina for formatada, nada se perde.**

## Antes de qualquer coisa: não abrir PR

O upstream **recusou esta feature por escrito**. A issue #157 está fechada e o
PR #284 (*"Add local Whisper voice transcripts"*, de outro autor, 2.300 linhas)
foi fechado em 29/09/2026 com este motivo:

> "Having looked at what doing this properly means (a choice of models, GPU
> support on each platform, model downloads and memory management), it amounts
> to hosting speech models inside a WhatsApp client, which is more than ZapFast
> should carry. I'm going to decline local transcription for now."

Este trabalho existe para uso pessoal. **Não abrir pull request.** Se algum dia
a decisão mudar, o caminho aceito é o PR #324 (inferência fora do processo, num
binário externo), não este.

## Sincronizar com o upstream

O comando diário, custo zero, ** rode mesmo sem precisar**:

```sh
git fetch upstream
git log --oneline HEAD..upstream/main   # o que o dono mexeu
```

Se aparecer algo nos seus arquivos (`app.rs`, `ui/conversation.rs`,
`ui/settings.rs`, `settings.rs`, `model.rs`, `demo.rs`), traga para dentro:

```sh
git rebase upstream/main
```

Se der ruim, `git rebase --abort` volta ao estado anterior sem perder nada.
Existe também a tag `backup-before-rebase` como rede extra.

### O conflito que vai acontecer: `pt-BR.po`

O rebase de 29/10/2026 deu conflito em **um arquivo só**, o catálogo de
traduções, e o código aplicou limpo nos outros dois commits.

**O lado `HEAD` desse conflito costuma estar vazio** (o dono acrescenta as
entradas dele *antes* do seu bloco), então a resolução é ficar com o lado do
seu commit:

```sh
# RUIM: descarta o lado do dono
git checkout --theirs assets/i18n/pt-BR.po

# CERTO: mantém tudo do dono E o seu bloco
```

Ao resolver, **confirme os dois lados**. Verificar só "não removi nada do
upstream" não basta — foi exatamente esse o erro que cometi: o arquivo ficou
sem nenhuma entrada de transcrição e a UI passou a falar inglês, com testes
verdes e build ok. Confira o seu:

```sh
grep -c 'msgid "Extensions"\|msgid "Delete model"' assets/i18n/pt-BR.po   # tem que ser 2
git diff upstream/main -- assets/i18n/pt-BR.po | grep -c '^-[^-]'        # tem que ser 0
```

### Depois de rebasar, sempre

```sh
cargo fmt --all --check
cargo clippy --locked --all-targets -- -D warnings
cargo test  --locked --all-targets
```

Referência do que passou no último rebase: fmt limpo, clippy limpo,
**1114 testes ok, 0 falhas**.

## Ambiente de build nesta máquina

O `whisper-rs` compila o whisper.cpp com CMake, e o projeto já traz
`openssl-sys` como vendorizado. Duas coisas precisam estar no ambiente:

```sh
export PATH="/c/Strawberry/perl/bin:$PATH"        # perl do Windows, não o do Git Bash
export CMAKE_GENERATOR=Ninja                      # o CMake 4.4 não acha gerador para o VS 18
export OPENSSL_DIR="C:\Program Files\OpenSSL"
export OPENSSL_LIB_DIR="C:\Program Files\OpenSSL\lib"
export OPENSSL_INCLUDE_DIR="C:\Program Files\OpenSSL\include"
```

O **Ninja** foi baixado de `ninja-build/ninja` (v1.12.1) para
`~/.cargo/bin/ninja.exe`. O crate `ninja` do crates.io é um stub e **não
serve**. Sem ele, o CMake falha com *"could not find any instance of Visual
Studio"*.

Sem essas variáveis o build falha em `openssl-sys` (perl) ou em
`whisper-rs-sys` (gerador do CMake). Nenhum dos dois é problema do código.

## Rodar na máquina

O build de dev usa **o mesmo diretório de dados do ZapFast instalado** —
`ProjectDirs::from("me", "paolino", …)` resolve para
`%LOCALAPPDATA%\paolino\zapfast\data` em debug e em release. Então a build
própria abre a mesma conta e os mesmos chats. Não é um executável isolado.

```sh
cargo build --release
target\release\zapfast.exe
```

### ⚠️ Desligue a atualização automática

`check_for_updates` vem `true` por padrão. Se você clicar em atualizar, o
`fastframe-update` baixa do `crmne/zapfast` e **substitui o seu executável pelo
oficial, silenciosamente**. Em `%APPDATA%\paolino\zapfast\config\settings.json`:

```json
"check_for_updates": false,
"download_updates_automatically": false
```

### Instalador de verdade (opcional)

O projeto usa Inno Setup e o script já existe em `packaging/windows/zapfast.iss`
— é o mesmo do CI. Saída **sem assinatura**, então o Windows mostra SmartScreen.

```sh
cargo build --release
& "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe" `
  "/DVersion=0.19.0" "/DNumericVersion=0.19.0" "/DArch=x64" `
  "/DBinary=$PWD\target\release\zapfast.exe" "/DOutputDir=$PWD\dist" `
  packaging\windows\zapfast.iss
```

Sai `dist\zapfast-v0.19.0-x64-pc-windows-msvc-setup.exe`. Não instale por cima
do ZapFast oficial: os dois escrevem em `%LOCALAPPDATA%\Programs\ZapFast` e só
um fica de pé.

No Linux, `packaging/install-user.sh target/release/zapfast` faz o equivalente.
DEB/RPM/AppImage/Flatpak/DGM não valem o esforço para uso pessoal: exigem
`native-packages` (gem Ruby), nFPM, `mksquashfs` e, no macOS, um Mac com
`codesign`.

## Medidas reais de desempenho (i5-13450HX, clip de 13,5 s)

| Modelo | Frio | Quente | Qualidade |
| --- | --- | --- | --- |
| Base (141 MB) | 2,1 s | 1,9 s | inútil |
| Small (466 MB) | 11,6 s | 7,6 s | erra quase tudo |
| Turbo (1,6 GB) | 28,0 s | **25,2 s** | quase perfeito |

Carregar o modelo custa ~3 s; o resto é inferência. Release quase não melhora
sobre debug porque o `whisper-rs-sys` força `CMAKE_BUILD_TYPE=RelWithDebInfo`.
**Turbo é ~2× mais lento que tempo real** — os modelos que cabem em CPU são
ruins, e o bom não acompanha o áudio. Foi medido com voz sintética: teste com
mensagens reais antes de concluir alguma coisa.

Modelos baixados em `C:\Users\teco_\zapfast-turbo-test\` (2,2 GB — pode apagar).

## Capturas

```sh
cargo run --features demo -- --demo --demo-page transcribed
cargo run --features demo -- --demo --demo-shot "C:\Users\teco_\zapfast-shots\x.png" --demo-page transcribed,dark
```

Cenários: `transcribed`, `transcribe-failed`. Combine com vírgula para
acumular (`transcribed,select`). Capturas prontas em
`C:\Users\teco_\zapfast-shots\rebase\`.

O `--demo-shot` com tema precisa ser conferido depois de qualquer mudança de
layout, porque foi olhando a imagem que apareceu um bug que 1114 testes não
pegaram.

## Pendência conhecida

`.github/scripts/update-translations.sh --check` **não roda nesta máquina**:
exige xgettext **0.24+** e o mais novo em qualquer repositório Ubuntu é 0.23.2,
que responde `language 'Rust' unknown`. Só importa se um dia abrir PR — o
resto da bateria passa. Quando rodar, ele reescreve os `.po` do jeito certo e
substitui as edições manuais do `pt-BR.po`.

## Dois bugs reais já corrigidos aqui

Guardei porque são fáceis de reintroduzir:

- **Resampling.** A versão original assumia 48 kHz e tirava a média de 3 em 3
  samples. Isso só funciona para OGG de voz; num anexo a 22.050 Hz daria
  7.350 Hz, lixo para o Whisper. Hoje há `resample()` usando a taxa real.
- **No-op silencioso.** Mensagem que chegasse antes do modelo terminar de
  baixar nunca era transcrita. Hoje os clipes ficam pendentes e são processados
  quando um download assenta — por contador, não por frame, para não pagar I/O
  de disco ocioso.
