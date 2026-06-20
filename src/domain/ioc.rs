//! Dependency Injection and IoC Container
//!
//! Provides a simple dependency injection container for managing service lifetimes
//! and dependencies, supporting the Dependency Inversion Principle.

use std::sync::Arc;
use anyhow::Result;

use crate::domain::{
    text_recognition::TextRecognitionService,
    language_processing::LanguageProcessingService,
    image_processing::ImageProcessingService,
    output_formatting::OutputFormattingService,
    config::ConfigurationService,
};

/// Service container for dependency injection
pub struct ServiceContainer {
    text_recognition: Option<Arc<TextRecognitionService>>,
    language_processing: Option<Arc<LanguageProcessingService>>,
    image_processing: Option<Arc<ImageProcessingService>>,
    output_formatting: Option<Arc<OutputFormattingService>>,
    config: Option<Arc<ConfigurationService>>,
}

impl ServiceContainer {
    pub fn new() -> Self {
        Self {
            text_recognition: None,
            language_processing: None,
            image_processing: None,
            output_formatting: None,
            config: None,
        }
    }

    /// Get or create the text recognition service
    pub fn text_recognition(&mut self) -> Result<Arc<TextRecognitionService>> {
        if let Some(service) = &self.text_recognition {
            return Ok(service.clone());
        }

        let builder = crate::domain::text_recognition::TextRecognitionBuilder::new();
        
        let service = Arc::new(builder.build()?);
        self.text_recognition = Some(service.clone());
        Ok(service)
    }

    /// Get or create the language processing service
    pub fn language_processing(&mut self) -> Arc<LanguageProcessingService> {
        if let Some(service) = &self.language_processing {
            return service.clone();
        }

        let service = Arc::new(LanguageProcessingService::new());
        self.language_processing = Some(service.clone());
        service
    }

    /// Get or create the image processing service
    pub fn image_processing(&mut self) -> Arc<ImageProcessingService> {
        if let Some(service) = &self.image_processing {
            return service.clone();
        }

        let service = Arc::new(ImageProcessingService::new());
        self.image_processing = Some(service.clone());
        service
    }

    /// Get or create the output formatting service
    pub fn output_formatting(&mut self) -> Arc<OutputFormattingService> {
        if let Some(service) = &self.output_formatting {
            return service.clone();
        }

        let service = Arc::new(OutputFormattingService::new());
        self.output_formatting = Some(service.clone());
        service
    }

    /// Get or create the configuration service
    pub fn config(&mut self) -> Arc<ConfigurationService> {
        if let Some(service) = &self.config {
            return service.clone();
        }

        let service = Arc::new(ConfigurationService::new());
        self.config = Some(service.clone());
        service
    }

    /// Set a custom text recognition service
    pub fn set_text_recognition(&mut self, service: Arc<TextRecognitionService>) {
        self.text_recognition = Some(service);
    }

    /// Set a custom language processing service
    pub fn set_language_processing(&mut self, service: Arc<LanguageProcessingService>) {
        self.language_processing = Some(service);
    }

    /// Set a custom image processing service
    pub fn set_image_processing(&mut self, service: Arc<ImageProcessingService>) {
        self.image_processing = Some(service);
    }

    /// Set a custom output formatting service
    pub fn set_output_formatting(&mut self, service: Arc<OutputFormattingService>) {
        self.output_formatting = Some(service);
    }

    /// Set a custom configuration service
    pub fn set_config(&mut self, service: Arc<ConfigurationService>) {
        self.config = Some(service);
    }
}

impl Default for ServiceContainer {
    fn default() -> Self {
        Self::new()
    }
}

/// Global service container (using thread-local for single-threaded scenarios)
thread_local! {
    static GLOBAL_CONTAINER: std::cell::RefCell<ServiceContainer> = std::cell::RefCell::new(ServiceContainer::new());
}

/// Get the global service container
pub fn global_container() -> &'static std::cell::RefCell<ServiceContainer> {
    &GLOBAL_CONTAINER
}

/// Reset the global service container (useful for testing)
pub fn reset_global_container() {
    GLOBAL_CONTAINER.with(|container| {
        *container.borrow_mut() = ServiceContainer::new();
    });
}
