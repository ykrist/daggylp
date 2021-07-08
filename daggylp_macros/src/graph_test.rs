#[macro_use]
use crate::*;
use quote::{ToTokens, TokenStreamExt, quote, quote_spanned};
use proc_macro2::{TokenStream, Span};
use syn::parse::{Parse, Parser, Result, ParseStream};
use syn::punctuated::Punctuated;
use syn::*;
use std::ops::Deref;


#[derive(Debug, Clone)]
pub enum GraphTestDirective {
  Config(VizConfig),
  Input(GraphTestInput),
}

#[derive(Debug, Clone)]
pub struct GraphTestInput {
  name: LitStr,
  meta: Option<TokenStream>,
}

impl Parse for GraphTestInput {
  fn parse(input: ParseStream) -> Result<Self> {
    let name = input.parse()?;
    let meta = if input.lookahead1().peek(Token![,]) {
      let _ : Token![,] = input.parse()?;
      let meta : TokenStream = input.parse()?;
      Some(meta)
    } else {
      None
    };
    Ok(GraphTestInput { name, meta })
  }
}

impl ToTokens for GraphTestInput {
  fn to_tokens(&self, tokens: &mut TokenStream) {
    let name = &self.name;
    if let Some(meta) = &self.meta {
      tokens.extend(quote! {
        (#name, #meta)
      });
    } else {
      name.to_tokens(tokens);
    }
  }
}

impl Directive for GraphTestDirective {
  fn from_attribute(attr: &Attribute) -> Result<Option<Self>> {
    let val = if attr.path.is_ident("config") {
      Some(GraphTestDirective::Config(attr.parse_args()?))
    } else if attr.path.is_ident("input") {
      Some(GraphTestDirective::Input(attr.parse_args()?))
    } else {
      None
    };
    Ok(val)
  }
}


impl ToTokens for GraphTestDirective {
  fn to_tokens(&self, tokens: &mut TokenStream) {
    use GraphTestDirective::*;
    match self {
      Input(name) => {
        tokens.extend(quote! {
          runner.run(#name);
        })
      }
      Config(config) =>
        config.to_tokens(tokens),
    }
  }
}


pub type GraphTest = Test<GraphTestDirective, ()>;

impl Parse for GraphTest {
  fn parse(input: ParseStream) -> Result<Self> {
    Test::parse(input, "GraphTestResult")
  }
}

impl ToTokens for GraphTest {
  fn to_tokens(&self, tokens: &mut TokenStream) {
    let GraphTest {
      outer_attrs,
      directives,
      fn_ident,
      fn_sig,
      fn_wrapper,
      fn_body,
      ..
    } = self;
    let inner_ident = &fn_sig.ident;

    let t = quote! {
      #[test]
      fn #fn_ident() {
        #(#outer_attrs)*
        #fn_sig
        #fn_body

        let mut runner = crate::test_utils::GraphTestRunner::new(
          crate::test_utils::get_test_id(module_path!(), stringify!(#fn_ident)),
          #fn_wrapper(#inner_ident)
        );

        #(#directives)*
      }
    };
    tokens.extend(t);
  }
}

