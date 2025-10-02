# Bug Fix: Typer Literal Type Issue

## Problema

```
RuntimeError: Type not yet supported: typing.Literal['rules_only', 'qa_only', 'fuse']
```

Errore durante l'esecuzione di `robimb` causato dall'uso di `typing.Literal` in un parametro Typer.

## Causa

Typer (la libreria CLI usata) non supporta nativamente `typing.Literal` come type hint per i parametri.

**File**: `src/robimb/cli/extract.py:166`

```python
# PRIMA (non funziona)
fusion: Literal["rules_only", "qa_only", "fuse"] = typer.Option(
    "fuse",
    "--fusion",
    case_sensitive=False,
    help="Fusion strategy between rules and QA",
),
```

## Soluzione

Cambiato il type hint da `Literal` a `str` e aggiunta validazione manuale.

**File**: `src/robimb/cli/extract.py`

```python
# DOPO (funziona)
fusion: str = typer.Option(
    "fuse",
    "--fusion",
    case_sensitive=False,
    help="Fusion strategy between rules and QA (choices: rules_only, qa_only, fuse)",
),

# Aggiunta validazione all'inizio della funzione
def extract_properties_command(...) -> None:
    # Validate fusion parameter
    valid_fusion_values = ["rules_only", "qa_only", "fuse"]
    if fusion not in valid_fusion_values:
        typer.echo(f"Error: --fusion must be one of: {', '.join(valid_fusion_values)}", err=True)
        raise typer.Exit(1)

    # ... resto del codice
```

## Verifica

```bash
# Test che ora funziona
robimb --help
robimb extract --help
robimb extract properties --help
```

Output atteso: help senza errori

## Note

- Questa è una limitazione nota di Typer (versione < 0.12)
- Alternative future:
  - Usare `click.Choice` direttamente
  - Aggiornare Typer quando supporterà Literal
  - Usare enum.Enum invece di Literal

## File Modificati

- `src/robimb/cli/extract.py` (linee 166-180)

## Test

```bash
# Verifica che il parametro funzioni correttamente
robimb extract properties --help | grep fusion

# Output atteso:
# --fusion   TEXT   Fusion strategy between rules and QA (choices: rules_only, qa_only, fuse) [default: fuse]
```
