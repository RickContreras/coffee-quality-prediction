"""
Script para descargar automáticamente los datos del Coffee Quality Institute desde Kaggle.

Este módulo gestiona la descarga automática del dataset de calidad del café
desde Kaggle, leyendo las credenciales desde un archivo kaggle.json local.

Uso:
    python scripts/download_data.py

Requiere:
    - Archivo kaggle.json en la raíz del proyecto con credenciales válidas
    - Librería kaggle instalada (pip install kaggle)
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

from kaggle.api.kaggle_api_extended import KaggleApi


# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class KaggleDataDownloader:
    """Gestor de descarga de datos desde Kaggle."""
    
    DATASET_NAME = "volpatto/coffee-quality-database-from-cqi"
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Inicializa el descargador de datos.
        
        Args:
            project_root: Ruta raíz del proyecto. Si es None, se detecta automáticamente.
        """
        if project_root is None:
            # Detectar raíz del proyecto (dos niveles arriba del script)
            self.project_root = Path(__file__).parent.parent
        else:
            self.project_root = project_root
            
        self.raw_data_path = self.project_root / "data" / "raw"
        self.kaggle_json_path = self.project_root / "kaggle.json"
        self.api: Optional[KaggleApi] = None
    
    def _verify_credentials_file(self) -> bool:
        """
        Verifica que el archivo kaggle.json exista en la raíz del proyecto.
        
        Returns:
            True si el archivo existe, False en caso contrario.
        """
        if not self.kaggle_json_path.exists():
            logger.error("No se encontró el archivo kaggle.json en la raíz del proyecto")
            logger.info(f"Ruta esperada: {self.kaggle_json_path}")
            logger.info("\nPasos para configurar:")
            logger.info("  1. Inicia sesión en https://www.kaggle.com/")
            logger.info("  2. Ve a Settings > API > Create New API Token")
            logger.info("  3. Descarga el archivo kaggle.json")
            logger.info("  4. Colócalo en la raíz del proyecto (junto al README.md)")
            return False
        return True
    
    def _authenticate(self) -> bool:
        """
        Configura y autentica la API de Kaggle usando credenciales locales.
        
        Returns:
            True si la autenticación fue exitosa, False en caso contrario.
        """
        try:
            with open(self.kaggle_json_path, 'r', encoding='utf-8') as f:
                credentials = json.load(f)
            
            # Validar estructura del archivo
            required_keys = ['username', 'key']
            missing_keys = [key for key in required_keys if key not in credentials]
            
            if missing_keys:
                logger.error(f"Faltan las siguientes claves en kaggle.json: {', '.join(missing_keys)}")
                return False
            
            # Configurar API de Kaggle
            self.api = KaggleApi()
            self.api.username = credentials['username']
            self.api.key = credentials['key']
            self.api.authenticate()
            
            logger.info(f"Autenticación exitosa para el usuario: {credentials['username']}")
            return True
            
        except json.JSONDecodeError:
            logger.error("El archivo kaggle.json no tiene un formato JSON válido")
            return False
        except Exception as e:
            logger.error(f"Error al autenticar con Kaggle: {e}")
            return False
    
    def _list_downloaded_files(self) -> None:
        """Lista y muestra información de los archivos descargados."""
        files = [f for f in self.raw_data_path.glob("*") if f.is_file()]
        
        if not files:
            logger.warning("No se encontraron archivos descargados")
            return
        
        logger.info("\nArchivos disponibles:")
        total_size = 0
        for file in sorted(files):
            size_mb = file.stat().st_size / (1024 * 1024)
            total_size += size_mb
            logger.info(f"  • {file.name} ({size_mb:.2f} MB)")
        
        logger.info(f"\nTamaño total: {total_size:.2f} MB")
    
    def download(self) -> bool:
        """
        Descarga el dataset de calidad del café desde Kaggle.
        
        Returns:
            True si la descarga fue exitosa, False en caso contrario.
        """
        # Verificar credenciales
        if not self._verify_credentials_file():
            return False
        
        # Autenticar
        if not self._authenticate():
            return False
        
        # Crear directorio de destino
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        
        # Descargar dataset
        logger.info(f"\nDescargando dataset: {self.DATASET_NAME}")
        logger.info(f"Destino: {self.raw_data_path}")
        
        try:
            self.api.dataset_download_files(
                self.DATASET_NAME,
                path=str(self.raw_data_path),
                unzip=True,
                quiet=False
            )
            
            logger.info("\n✓ Descarga completada exitosamente")
            self._list_downloaded_files()
            return True
            
        except Exception as e:
            logger.error(f"Error al descargar datos: {e}")
            logger.info("\nVerifica que:")
            logger.info("  • Las credenciales en kaggle.json sean correctas")
            logger.info("  • Tengas acceso al dataset en Kaggle")
            logger.info("  • Hayas aceptado las reglas del dataset")
            logger.info(f"  • URL del dataset: https://www.kaggle.com/datasets/{self.DATASET_NAME}")
            return False


def main() -> int:
    """
    Función principal para ejecutar la descarga de datos.
    
    Returns:
        Código de salida: 0 si fue exitoso, 1 en caso contrario.
    """
    logger.info("=== Coffee Quality Data Downloader ===\n")
    
    downloader = KaggleDataDownloader()
    success = downloader.download()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())