use crate::*;
use proc_macro2::TokenStream;
use syn::*;
use syn::parse::{Parse, ParseStream, Parser};
use syn::punctuated::Punctuated;
use quote::{ToTokens, quote};
use std::ops::Deref;


#[derive(Default, Debug, Copy, Clone)]
pub struct GlobalSettings {
  deterministic: bool,
  skip_regressions: bool,
  debug: bool,
}

impl UpdateFromMeta for GlobalSettings {
  fn update(&mut self, meta: &Meta) -> Result<bool> {
    match meta {
      Meta::Path(p) => {
        if p.is_ident("skip_regressions") {
          self.skip_regressions = true;
        } else if p.is_ident("debug") {
          self.debug = true;
        } else if p.is_ident("deterministic") {
          self.deterministic = true;
        } else {
          return Ok(false);
        }
        Ok(true)
      }
      _ => Ok(false)
    }
  }
}

impl_parse_for_updatefrommeta!(GlobalSettings);
//
// impl Parse for GlobalSettings {
//   fn parse(input: ParseStream) -> Result<Self> {
//     let meta_list = input.call(Punctuated::<Meta, Token![,]>::parse_terminated)?;
//     let mut settings = GlobalSettings::default();
//     for meta in meta_list.iter() {
//       if settings.update(meta)? {
//         continue;
//       }
//       parse_error!(meta => "unrecognised attribute item")
//     }
//     Ok(settings)
//   }
// }
//

#[derive(Default, Debug, Clone)]
pub struct Config {
  viz: VizConfig,
  cpus: Option<LitInt>,
  cases: Option<LitInt>,
}

fn expect_litint(lit: &Lit) -> Result<&LitInt> {
  match lit {
    Lit::Int(l) => Ok(l),
    _ => parse_error!(lit => "expected integer literal"),
  }
}

fn expect_litbool(lit: &Lit) -> Result<&LitBool> {
  match lit {
    Lit::Bool(l) => Ok(l),
    _ => parse_error!(lit => "expected integer literal"),
  }
}

impl UpdateFromMeta for Config {
  fn update(&mut self, meta: &Meta) -> Result<bool> {
    if self.viz.update(meta)? {
      return Ok(true);
    }
    match meta {
      Meta::NameValue(kwarg) => {
        if kwarg.path.is_ident("cpus") {
          self.cpus = Some(expect_litint(&kwarg.lit)?.clone())
        } else if kwarg.path.is_ident("cases") {
          self.cases = Some(expect_litint(&kwarg.lit)?.clone())
        } else {
          return Ok(false);
        }
        Ok(true)
      }
      _ => Ok(false)
    }
  }
}

impl_parse_for_updatefrommeta!(Config);

impl ToTokens for Config {
  fn to_tokens(&self, tokens: &mut TokenStream) {
    let mut config_block = self.viz.to_token_stream();
    if let Some(n) = &self.cpus {
      config_block.extend(quote! {
        runner.cpus(#n);
      })
    }
    if let Some(n) = &self.cases {
      config_block.extend(quote! {
        runner.cases(#n);
      })
    }
    tokens.extend(quote! {{ #config_block }});
  }
}

#[derive(Debug, Clone)]
pub enum ProptestDirective {
  Config(Config),
  Input(TokenStream),
}

fn strip_parens(input: ParseStream) -> Result<TokenStream> {
  let content;
  parenthesized!(content in input);
  content.parse()
}

impl Directive for ProptestDirective {
  fn from_attribute(attr: &Attribute) -> Result<Option<Self>> {
    let val = if attr.path.is_ident("config") {
      Some(ProptestDirective::Config(attr.parse_args()?))
    } else if attr.path.is_ident("input") {
      let s = strip_parens.parse2(attr.tokens.clone())?;
      Some(ProptestDirective::Input(s))
    } else {
      None
    };
    Ok(val)
  }
}

impl ToTokens for ProptestDirective {
  fn to_tokens(&self, tokens: &mut TokenStream) {
    use ProptestDirective::*;
    match self {
      Input(ts) => {
        tokens.extend(quote! {
          runner.run(#ts);
        })
      }
      Config(config) =>
        config.to_tokens(tokens),
    }
  }
}

pub type GraphProptest = Test<ProptestDirective, GlobalSettings>;

impl Parse for GraphProptest {
  fn parse(input: ParseStream) -> Result<Self> {
    Test::parse(input, "GraphProptestResult")
  }
}

impl ToTokens for GraphProptest {
  fn to_tokens(&self, tokens: &mut TokenStream) {
    let GraphProptest {
      global_settings,
      outer_attrs,
      directives,
      fn_ident,
      fn_sig,
      fn_wrapper,
      fn_body
    } = self;
    let global_settings = global_settings.unwrap_or_default();

    let directives = if global_settings.debug {
      quote! { runner.debug(); }
    } else {
      let deterministic = if global_settings.deterministic {
        Some(quote! { runner.deterministic(true ); })
      } else {
        None
      };
      let skip_regressions = if global_settings.skip_regressions {
        Some(quote! { runner.skip_regressions(true); })
      } else {
        None
      };
      quote! {
        #skip_regressions
        #deterministic
        #(#directives)*
      }
    };

    let inner_ident = &fn_sig.ident;

    let t = quote! {
      #[test]
      fn #fn_ident() {
        #(#outer_attrs)*
        #fn_sig
        #fn_body

        let mut runner = crate::test_utils::GraphProptestRunner::new(
          crate::test_utils::get_test_id(module_path!(), stringify!(#fn_ident)),
          #fn_wrapper(#inner_ident)
        );

        #directives
      }
    };
    tokens.extend(t);
  }
}
